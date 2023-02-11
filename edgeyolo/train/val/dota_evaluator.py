import contextlib
import datetime
import io
import itertools
import json
from pycocotools.cocoeval import COCOeval
import time
from loguru import logger
from tqdm import tqdm

import torch
import torch.distributed


from ...detect.dotaDetector import DOTADetector
from ...utils import (
    gather,
    synchronize,
    time_synchronized,
    xyxy2xywh
)



class DOTAEvaluator:
    """
    DOTA AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, dataloader, img_size, confthre, nmsthre, num_classes, testdev=False, rank=0, obj_conf_enabled=True, save=False
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.rank = rank
        self.is_main_process = rank == 0
        self.obj_conf_enabled = obj_conf_enabled
        self.save = save

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor

        model = model.eval()
        if half:
            model = model.half()

        ids = []
        data_list = []

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        detector = DOTADetector(
            model,
            self.confthre,
            self.nmsthre,
            self.num_classes,
            self.obj_conf_enabled,
            half
        )

        for cur_iter in tqdm(range(len(self.dataloader)), ncols=100) if self.is_main_process else range(len(self.dataloader)):


            with torch.no_grad():

                imgs = self.dataset.load_image(cur_iter)
                _, _, _, img_id, _ = self.dataset.pull_item(cur_iter)
                img_id = int(img_id)

                # imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1

                outputs = detector(imgs)

                if is_time_record:
                    inference_time += detector.infer_time
                    nms_time += detector.nms_time

            data_list.extend(detector.result2coco_json(outputs, img_id))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        # print("distributed:", distributed)
        if distributed:
            try:
                if self.is_main_process:
                    logger.info("Rank0: gathering data from subprocess...")
                data_list = gather(data_list, dst=0)
                data_list = list(itertools.chain(*data_list))
                torch.distributed.reduce(statistics, dst=0)
                if self.is_main_process:
                    logger.info("data gathered.")
            except Exception as e:
                logger.error(e)

        if self.is_main_process and self.save:
            with open(f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_test.json", "w") as f:
                json.dump(data_list, f)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_prediction(self, data_dict, statistics):
        # logger.info(f"rank: {self.rank}")
        if not self.is_main_process and self.rank:
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco

            if self.testdev:
                json.dump(data_dict, open("./testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./testdev_2017.json")
            else:
                cocoDt = cocoGt.loadRes(data_dict)

            logger.info("Use standard COCO evaluator.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
