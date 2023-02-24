from .loss import YoloLoss, iou_loss, cls_loss, conf_loss
from .optimizer import get_optimizer
from .val import Evaluator, evaluators
from ..models import *
from ..utils import (
    ModelEMA,
    LRScheduler,
    all_reduce_norm,
    get_model_info,
    gpu_mem_usage,
    synchronize,
    is_parallel,
    NoPrint,
)

from ..data import (
    COCODataset,
    TrainTransform,
    MosaicDetection,
    EnhancedMosaicDetection,
    MaskDataLoader,
    preprocess,
    ValTransform,
    get_dataset,
)

import tabulate
import datetime
import yaml

import os.path as osp
import torch.utils.data
import torch.distributed as dist

import time

from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer(EdgeYOLO):

    dataloader: MaskDataLoader = None
    evaluator: Evaluator = None
    lr_scheduler: LRScheduler = None
    loss: YoloLoss = None
    ema_model: ModelEMA = None
    dataset_cfg: dict

    def __init__(self, params: dict, rank=0):
        with NoPrint(rank):
            nc = None
            names = None
            if params.get("dataset_cfg") is not None:
                with open(params.get("dataset_cfg"), "r") as f:
                    self.dataset_cfg: dict = yaml.load(f, yaml.SafeLoader)
                    names = self.dataset_cfg.get("names")
                    nc = len(names)

            super(Trainer, self).__init__(params["model_cfg"] if "model_cfg" in params else None,
                                          params["weights"],
                                          rank,
                                          params["use_cfg"],
                                          nc=nc)

            self.load_class_names(osp.join(params["dataset_dir"], params["class_names_file"])
                                  if names is None else names)

        self.params = params

        self.device = f'cuda:{params["device"][rank]}'
        torch.cuda.set_device(self.device)

        self.data_show = [[k, v] for k, v in params.items()]
        self.data_type = torch.float16 if self.params["fp16"] else torch.float32
        self.world_size = len(params["device"])

        self.max_epoch = self.params.get("max_epoch") or 300

        self.max_iter = 0
        self.now_iter = 0
        self.start_time = 0
        self.best_ap = 0.0
        self.best_epoch = -1
        self.no_aug = False
        self.print_data = {}
        self.is_distributed = self.world_size > 1

        self.eval_interval = self.params.get("eval_interval") or 1
        self.input_size = self.params.get("input_size") or (640, 640)

        self.params["multiscale_range"] = self.params.get("multiscale_range") or 5

        min_size = int(self.input_size[0] / 32) - self.params["multiscale_range"]
        max_size = int(self.input_size[0] / 32) + self.params["multiscale_range"]
        self.random_size = (min_size, max_size)

        self.eval_data = {}
        self.eval_file = os.path.join(self.params["output_dir"], "eval.yaml")
        if osp.isfile(self.eval_file) and self.rank == 0:
            self.eval_data = yaml.load(open(self.eval_file, "r", encoding="utf8").read(), yaml.Loader)
            self.best_ap = max([self.eval_data[key]["ap50_95"] for key in self.eval_data])

        self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.params["fp16"])

    def __repr__(self):
        return tabulate.tabulate(self.data_show, ["Keywords", "Values"], "fancy_grid")

    def load_init(self):

        """最耗时的加载"""

        def load_dataloader():
            with NoPrint(self.rank):
                if self.rank == 0:
                    logger.info("loading dataset...")

                dataset = get_dataset(
                    cfg=self.dataset_cfg,
                    img_size=self.input_size,
                    preproc=TrainTransform(
                        max_labels=200,
                        flip_prob=self.params["flip_prob"],
                        hsv_prob=1,
                        hsv_gain=self.params["hsv_gain"]
                    ),
                    mode="train",
                    save_cache=self.rank == 0
                )

            if not self.params["enhance_mosaic"]:
                dataset = MosaicDetection(
                    dataset,
                    mosaic=True,
                    img_size=self.input_size,
                    preproc=dataset.preproc,
                    degrees=float(self.params["degrees"]),
                    translate=0.1,
                    mosaic_scale=self.params["mosaic_scale"],
                    mixup_scale=self.params["mixup_scale"],
                    shear=2.0,
                    enable_mixup=self.params["enable_mixup"],
                    mosaic_prob=self.params["mosaic_prob"],
                    mixup_prob=self.params["mixup_prob"],
                    rank=self.rank,
                    train_mask=False
                )
            else:
                dataset = EnhancedMosaicDetection(
                    dataset,
                    mosaic=True,
                    img_size=self.input_size,
                    preproc=dataset.preproc,
                    degrees=float(self.params["degrees"]),
                    translate=0.1,
                    mosaic_scale=self.params["mosaic_scale"],
                    mixup_scale=self.params["mixup_scale"],
                    shear=2.0,
                    enable_mixup=self.params["enable_mixup"],
                    mosaic_prob=self.params["mosaic_prob"],
                    mixup_prob=self.params["mixup_prob"],
                    rank=self.rank,
                    train_mask=False,
                    n_mosaic=2
                )

            if self.rank == 0:
                logger.info("init data prefetcher...")
            self.dataloader = MaskDataLoader(
                dataset,
                batch_size=self.params["batch_size_per_gpu"],
                rank=self.rank,
                world_size=self.world_size,
                num_workers=self.params["loader_num_workers"],
                train_mask=False
            )
            if self.rank == 0:
                logger.info("prefetcher loaded!")
            self.max_iter = len(self.dataloader)

        def load_evaluator():
            # import torch.utils.data
            if self.rank == 0:
                logger.info("loading evaluator...")

            with NoPrint():
                valdataset = get_dataset(
                    cfg=self.dataset_cfg,
                    img_size=self.input_size,
                    preproc=ValTransform(legacy=False),
                    mode="val"
                )

            if self.is_distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
            else:
                sampler = torch.utils.data.SequentialSampler(valdataset)

            dataloader_kwargs = {
                "num_workers": self.params["loader_num_workers"],
                "pin_memory": True,
                "sampler": sampler,
                "batch_size": self.params["batch_size_per_gpu"]
            }
            val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

            self.evaluator = evaluators.get(self.dataset_cfg.get("type"))(
                dataloader=val_loader,
                img_size=self.input_size,
                confthre=self.params["val_conf_thres"],
                nmsthre=self.params["val_nms_thres"],
                num_classes=len(self.class_names),
                testdev=False,
                rank=self.rank,
                obj_conf_enabled=self.params["obj_conf_enabled"],
                save="save" in self.params and self.params["save"]
            )
            if self.rank == 0:
                logger.info("evaluator loaded.")

        def load_optimizer():
            if self.rank == 0:
                logger.info("loading optimizer...")
            self.optimizer = get_optimizer(
                model=self.model,
                lr=self.params["lr_per_img"],
                momentum=self.params["momentum"],
                weight_decay=self.params["weight_decay"],
                train_backbone=self.params["train_backbone"],
                head_layer_num=self.params["train_start_layers"],
                optimizer_type=self.params["optimizer"]
            )
            if self.is_match and self.params["load_optimizer_params"]:
                if hasattr(self, "ckpt") and "optimizer" in self.ckpt:
                    try:
                        self.optimizer.load_state_dict(self.ckpt["optimizer"])
                    except RuntimeError as e:
                        if self.rank == 0:
                            logger.error(e)
                    except ValueError:
                        if self.rank == 0:
                            logger.warning("optimizer params not match!, skip loading.")

        def load_loss():
            if self.rank == 0:
                logger.info("loading loss...")
            self.loss = YoloLoss(
                class_loss=cls_loss(self.params["loss_use"][0]),
                confidence_loss=conf_loss(self.params["loss_use"][1]),
                bbox_loss=iou_loss(self.params["loss_use"][2]),
                nc=self.model.yaml['nc'],
            )

        def load_lr_scheduler(**kwargs):
            if self.rank == 0:
                logger.info("init learning rate scheduler...")
            base_lr = self.params["lr_per_img"] * self.params["batch_size_per_gpu"] * self.world_size
            self.lr_scheduler = LRScheduler(
                "yolowarmcos",   # "yoloxsemiwarmcos"  "warmcos"   "cos"   "multistep"
                base_lr,
                self.max_iter,
                self.max_epoch,
                warmup_epochs=self.params["warmup_epochs"],
                warmup_lr_start=base_lr * self.params["warmup_lr_ratio"],
                no_aug_epochs=self.params["close_mosaic_epochs"],
                min_lr_ratio=self.params["final_lr_ratio"],
                **kwargs
            )

        if not self.params["eval_only"]:
            load_loss()
            load_optimizer()
            load_dataloader()
            load_lr_scheduler()
        load_evaluator()

    def random_resize(self):
        tensor = torch.LongTensor(2).to(device=self.device)

        if self.rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if self.is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        self.input_size = (tensor[0].item(), tensor[1].item())

    def count_eta(self, second_used):
        iter_already = self.progress_in_iter + 1 - self.start_epoch * self.max_iter
        iter_remain = self.max_epoch * self.max_iter - self.progress_in_iter - 1
        second_remain = second_used * iter_remain / iter_already
        return datetime.timedelta(seconds=int(second_remain))

    def update_print_data(self, **kwargs):
        self.print_data = {
            "epoch": "%d/%d" % (self.now_epoch + 1, self.max_epoch),
            "iter": "%d/%d" % (self.now_iter + 1, self.max_iter),
            "mem": f"{int(gpu_mem_usage())}MB"
        }
        losses = {}
        for key in kwargs:
            if key == "l1_loss" and not self.loss.use_l1:
                continue
            if key.endswith("_loss"):
                losses[key[:-5]] = kwargs[key]
                continue

            if not key == "data_time":
                self.print_data[key] = kwargs[key]

        loss_str = ""
        for k, v in losses.items():
            loss_str += f"{k}:{v:.2f} "
        loss_str = "{%s}" % loss_str[:-1]
        self.print_data["loss"] = loss_str

        eta = self.count_eta(time.time()-self.start_time)
        self.print_data["ETA"] = str(eta)

    @property
    def progress_in_iter(self):
        return self.now_epoch * self.max_iter + self.now_iter

    def train(self):

        def before_train():
            if self.rank == 0:
                logger.info(f"params:\n{self}")
                logger.info(f"num classes: {self.model.yaml['nc']}")
                logger.info(get_model_info(self.model, self.input_size))

            self.load_init()

            if self.is_distributed:
                self.model = DDP(self.model, device_ids=[self.params["device"][self.rank]], broadcast_buffers=False)

            if self.params["use_ema"]:
                self.ema_model = ModelEMA(self.model, 0.9998)
                self.ema_model.updates = self.max_iter * self.start_epoch

            os.makedirs(self.params["output_dir"], exist_ok=True)

            if self.params["force_start_epoch"] >= 0:
                self.start_epoch = self.params["force_start_epoch"]
                if self.rank == 0:
                    logger.info(f"training is forced to start at epoch {self.start_epoch + 1}")

            if self.rank == 0:
                logger.info("Training start...")
                if self.start_epoch == 0:
                    logger.info("\n{}".format(self.model))

            if self.params["eval_at_start"]:
                self.now_epoch = max(0, self.start_epoch - 1)
                self.evaluate()

            self.start_time = time.time()

        def train_one_epoch():
            def before_epoch():
                epoch_info = "Start Train Epoch %d " % (self.now_epoch + 1)

                if self.now_epoch >= self.max_epoch - self.params["close_mosaic_epochs"] or self.no_aug:
                    epoch_info += "(No mosaic aug, L1 loss enabled)"
                    self.dataloader.close_mosaic()

                    self.loss.use_l1 = True

                    if not self.no_aug:
                        if self.rank == 0:
                            self.save_ckpt(filename="last_augmentation_epoch.pth")
                        self.no_aug = True
                if self.rank == 0:
                    logger.info(epoch_info)
                    # for debug
                    # self.save_ckpt(file_name="start.pth")

            def train_one_iter():
                def before_iter():
                    self.random_resize()

                def train_in_iter():
                    iter_start_time = time.time()

                    inps, targets = self.dataloader.next()
                    mask_edge = None

                    inps = inps.to(self.data_type)
                    targets = targets.to(self.data_type)
                    targets.requires_grad = False

                    inps, targets = preprocess(inps, targets, self.input_size)
                    data_end_time = time.time()

                    with torch.cuda.amp.autocast(enabled=self.params["fp16"]):
                        while True:
                            try:
                                outputs = self.model(inps)
                                break
                            except RuntimeError:
                                logger.warning(f"{self.device} out of memory, try again")
                                torch.cuda.empty_cache()
                                time.sleep(1)

                    outputs = self.loss(outputs, (targets, mask_edge))
                    loss = outputs["total_loss"]

                    self.optimizer.zero_grad()
                    while True:
                        try:
                            self.scaler.scale(loss).backward()
                            break
                        except RuntimeError:
                            logger.warning(f"{self.device} out of memory, try again")
                            torch.cuda.empty_cache()
                            time.sleep(1)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.params["use_ema"]:
                        self.ema_model.update(self.model)

                    lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr

                    iter_end_time = time.time()
                    if (self.now_iter + 1) % self.params["print_interval"] == 0:
                        self.update_print_data(
                            t_iter=iter_end_time - iter_start_time,
                            # data_time=data_end_time - iter_start_time,
                            lr=lr,
                            **outputs,
                        )

                def after_iter():
                    if (self.now_iter + 1) % self.params["print_interval"] == 0:
                        iter_info = ""
                        for key in self.print_data:
                            if isinstance(self.print_data[key], int):
                                iter_info += "%s:%d " % (key, self.print_data[key])
                            elif isinstance(self.print_data[key], str):
                                iter_info += "%s:%s " % (key, self.print_data[key])
                            else:
                                if key == "lr":
                                    iter_info += "%s:%.3e " % (key, self.print_data[key])
                                else:
                                    iter_info += "%s:%.2f " % (key, self.print_data[key])
                        iter_info = iter_info[:-1]
                        # torch.cuda.empty_cache()
                        if self.rank == 0:
                            logger.info(iter_info)

                before_iter()
                train_in_iter()
                after_iter()

            def after_epoch():
                if self.rank == 0:
                    ckpt_name = "last.pth"
                    self.save_ckpt(ckpt_name)
                    if self.params["save_checkpoint_for_each_epoch"]:
                        ckpt_name = f"epoch_{str(self.now_epoch + 1).zfill(len(str(self.max_epoch)))}.pth"
                        self.save_ckpt(ckpt_name)

                if self.is_distributed:
                    all_reduce_norm(self.model, self.world_size)
                torch.cuda.empty_cache()
                if (self.now_epoch + 1) % self.eval_interval == 0 or self.no_aug:
                    while True:
                        try:
                            self.evaluate()
                            break
                        except Exception as e:
                            logger.error(f"error: {e}")
                            torch.cuda.empty_cache()

            before_epoch()
            for self.now_iter in range(self.max_iter):
                train_one_iter()
            after_epoch()

        def after_train():
            if self.rank == 0:
                logger.info("Training Finished.")
                logger.info("Time Spent: %s" % str(datetime.timedelta(seconds=(time.time() - self.start_time))))
                logger.info(f"best mAP_50:95 = {self.best_ap:.5f} at epoch {self.best_epoch + 1}.")

        if self.params["eval_only"]:
            self.evaluate_only()
        else:
            before_train()
            for self.now_epoch in range(self.start_epoch, self.max_epoch):
                train_one_epoch()
            after_train()

    def evaluate(self):
        if self.params["use_ema"]:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        ap50_95, ap50, summary = self.evaluator.evaluate(
            evalmodel, self.is_distributed
        )
        if self.params["eval_only"]:
            if self.rank == 0:
                logger.info("\n" + summary)
            return ap50_95, ap50, summary
        eval_msg = {
            "ap50_95": float(ap50_95),
            "ap50": float(ap50),
        }
        self.model.train()
        if self.rank == 0:
            logger.info("\n" + summary)
        synchronize()

        pth_save = ["last.pth"]
        if ap50_95 > self.best_ap:
            pth_save.append("best.pth")
            self.best_epoch = self.now_epoch
        self.best_ap = max(self.best_ap, ap50_95)

        if self.params["save_checkpoint_for_each_epoch"]:
            pth_save.append(f"epoch_{str(self.now_epoch + 1).zfill(len(str(self.max_epoch)))}.pth")

        if self.rank == 0:
            self.save_ckpt(pth_save, **eval_msg)
            self.eval_data[self.now_epoch] = eval_msg
            yaml.dump(self.eval_data, open(self.eval_file, "w", encoding="utf8"))

    def evaluate_only(self, file_list=None, save=True):
        from glob import glob

        if self.rank == 0:
            logger.info(f"params:\n{self}")

        torch.cuda.set_device(self.device)

        self.load_init()
        if isinstance(file_list, str):
            file_list = [file_list]
        elif file_list is None:
            file_list = sorted(glob(os.path.join(self.params["output_dir"], "*.pth")))
        for self.params["weights"] in file_list:
            self.params["weights"] = self.params["weights"].replace("\\", "/")
            if self.rank == 0:
                logger.info("now evaluate: " + self.params["weights"])

            self.try_load_state_dict(self.params["weights"])
            try:
                self.model.to(self.device)
            except Exception as e:
                logger.error(e)

            if self.is_distributed:
                self.model = DDP(self.model, device_ids=[self.params["device"][self.rank]], broadcast_buffers=False)
            if self.params["use_ema"]:
                self.ema_model = ModelEMA(self.model, 0.9998)
                self.ema_model.updates = self.max_iter * self.start_epoch

            ap50, ap50_95, summary = self.evaluate()
            if save:
                result_file_name = self.params["weights"].split("/")[-1].split(".")[0] + ".txt"
                if self.rank == 0:
                    output_dir = os.path.join(self.params["output_dir"], "eval_result")
                    os.makedirs(output_dir, exist_ok=True)
                    with open(os.path.join(output_dir, result_file_name), "w") as f:
                        f.write(f"ap50: {ap50}\nap50_95: {ap50_95}\n{summary}")
            if len(file_list) == 1:
                return ap50, ap50_95

    def save_ckpt(self, filename, **kwargs):
        filename = [filename] if isinstance(filename, str) else filename
        save_path = [osp.join(self.params["output_dir"], fn) for fn in filename]

        model_save = self.ema_model.ema \
            if self.params["use_ema"] else self.model.module \
            if self.is_distributed else self.model
        kwargs["model"] = model_save.state_dict()

        self.save(save_path, kwargs)
