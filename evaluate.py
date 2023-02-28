import os
import os.path

import torch
import cv2

import torch.multiprocessing as mp
from torch import distributed as dist
import torch.backends.cudnn as cudnn
from datetime import timedelta
from loguru import logger
from edgeyolo.train.trainer import Trainer as Evaluator


def get_args():
    import argparse

    parser = argparse.ArgumentParser("EdgeYOLO evaluate parser")
    parser.add_argument("-w", "--weights", type=str, default="edgeyolo_coco.pth", help="weights")
    parser.add_argument("-b", "--batch", type=int, default=8, help="batch size for each device")
    parser.add_argument("-i", "--input-size", type=int, nargs="+", default=[640, 640], help="image input size")

    parser.add_argument("--dataset", type=str, default="params/dataset/coco.yaml", help="dataset config")
    parser.add_argument("--device", type=int, nargs="+", default=[0], help="eval device")

    parser.add_argument("--no-obj-conf", action="store_true", help="for debug only, do not use it")
    parser.add_argument("--save", action="store_true", help="save deploy model without optimizer params")
    parser.add_argument('--fp16', action='store_true', help='half precision')
    parser.add_argument('--trt', action='store_true', help='is tensorrt model')
    return parser.parse_args()


def generate_params(**kwargs):
    PARAMS = {
        "dataset_cfg": "params/dataset/coco.yaml",
        "input_size": [640, 640],
        "weights": "edgeyolo_coco.pth",
        "device": [0, 1, 2, 3],
        "val_conf_thres": 0.001,
        "val_nms_thres": 0.65,
        "num_threads": 1,
        "batch_size_per_gpu": 8,
        "loader_num_workers": 4,
        "eval_only": True,
        "cudnn_benchmark": True,
        "fp16": False,
        "multiscale_range": 0,
        "output_dir": "eval_output",
        "use_ema": False,
        "use_cfg": False,
        "obj_conf_enabled": True,
        "save": False,
        "trt": False,
        "pixel_range": 255
    }
    for k, v in kwargs.items():
        PARAMS[k] = v
    return PARAMS


def load_evaluator(rank, params, is_distributed):
    # import torch.utils.data
    from edgeyolo import NoPrint
    from edgeyolo.data import get_dataset, ValTransform
    from edgeyolo.train.val import evaluators

    if rank == 0:
        logger.info("loading evaluator...")

    dataset_cfg = params.get("dataset_cfg")
    with NoPrint():
        valdataset, dataset_type = get_dataset(
            cfg=dataset_cfg,
            img_size=params.get("input_size"),
            preproc=ValTransform(legacy=params["pixel_range"] == 1),
            mode="val",
            get_type=True
        )

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {
        "num_workers": params["loader_num_workers"],
        "pin_memory": True,
        "sampler": sampler,
        "batch_size": params["batch_size_per_gpu"]
    }
    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    evaluator = evaluators.get(dataset_type)(
        dataloader=val_loader,
        img_size=params["input_size"],
        confthre=params["val_conf_thres"],
        nmsthre=params["val_nms_thres"],
        num_classes=len(valdataset.names),
        testdev=False,
        rank=rank,
        obj_conf_enabled=params["obj_conf_enabled"],
        save="save" in params and params["save"]
    )
    if rank == 0:
        logger.info("evaluator loaded.")
    return evaluator

def eval_single(
    rank=0,
    params=None,
    dist_url="tcp://127.0.0.1:12345"
):
    torch.set_num_threads(params["num_threads"])
    cv2.setNumThreads(params["num_threads"])

    params["device"] = params["device"] if isinstance(params["device"], list) else [params["device"]]
    device = params["device"][rank]
    torch.cuda.set_device(device)

    world_size = len(params["device"])
    if world_size > 1:
        try:
            dist.init_process_group(
                backend="gloo",
                init_method=dist_url,
                world_size=world_size,
                rank=rank,
                timeout=timedelta(minutes=30),
            )
        except Exception:
            logger.error("Process group URL: {}".format(dist_url))
            raise

        dist.barrier()
        cudnn.benchmark = params["cudnn_benchmark"]


    ey = None
    if params.get("trt"):
        # os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        from edgeyolo.detect import TRTDetector
        detector = TRTDetector(params.get("weights"), 0, 0)
        model = detector.model
        params["batch_size_per_gpu"] = detector.batch_size
        params["input_size"] = detector.input_size
    else:
        from edgeyolo import EdgeYOLO
        ey = EdgeYOLO(weights=params.get("weights"))
        model = ey.model
        params["pixel_range"] = ey.ckpt.get("pixel_range") or 255

    model.cuda(params["device"][rank])

    evaluator = load_evaluator(rank, params, world_size > 1)
    ap50_95, ap50, summary = evaluator.evaluate(
        model, world_size > 1, params.get("fp16") or False
    )
    logger.info(f"ap50    : {ap50}")
    logger.info(f"ap50_95 : {ap50_95}")
    logger.info(summary)
    # ap50, ap50_95 = evaluator.evaluate_only(params["weights"], False)

    if params.get("save") and ey is not None:
        ey.ckpt.pop("optimizer") if "optimizer" in ey.ckpt.keys() else None
        ey.ckpt["ap50"] = ap50
        ey.ckpt["ap50_95"] = ap50_95
        ey.ckpt["epoch"] = -1
        logger.info(f"\nap50:95 = {ap50_95}\n"
                    f"ap50    = {ap50}")
        path, f = os.path.dirname(params["weights"]), os.path.basename(params["weights"])
        filepath = f"eval_{f}"
        torch.save(ey.ckpt, os.path.join(path, filepath))
        logger.info(f"deploy model saved to {filepath}")


def launch(params):
    world_size = len(params["device"])
    is_distributed = world_size > 1

    if is_distributed:

        def find_free_port():
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", 0))
            port = sock.getsockname()[1]
            sock.close()
            return port

        dist_url = f"tcp://127.0.0.1:{find_free_port()}"
        start_method = "spawn"

        mp.start_processes(
            eval_single,
            nprocs=world_size,
            args=(
                params,
                dist_url
            ),
            daemon=False,
            start_method=start_method,
        )
    else:
        eval_single(params=params)


if __name__ == '__main__':

    args = get_args()

    launch(
        generate_params(
            weights=args.weights,
            dataset_cfg=args.dataset,
            device=[*set(args.device)],
            batch_size_per_gpu=args.batch,
            input_size=args.input_size,
            obj_conf_enabled=not args.no_obj_conf,
            save=args.save,
            fp16=args.fp16,
            trt=args.trt
        )
    )
