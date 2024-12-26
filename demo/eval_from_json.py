# for deploy model (rknn, hbdnn, ascend, mnn, tensorrt)
"""
json format:

{
    "XXXXXX.jpg": {"category_id": int, "bbox": List[float](x1,y1,w,h), "score": float, "segmentation": []},
    ......
}

"""
import os
import os.path
import os.path as osp
import argparse
from loguru import logger

import sys
sys.path.append("./")

from edgeyolo.data import get_dataset, COCODataset


def get_args():
    parser = argparse.ArgumentParser("EdgeYOLO evaluate parser")
    parser.add_argument("-c", "--config", type=str, default="eval_results.json", help="weights")
    parser.add_argument("--dataset", type=str, default="params/dataset/coco.yaml", help="dataset config")
    return parser.parse_args()


def generate_params(**kwargs):
    PARAMS = {
        "dataset_cfg": "params/dataset/coco.yaml",
        "config": "eval_results.json",
    }
    for k, v in kwargs.items():
        PARAMS[k] = v
    return PARAMS


def load_dataset(params):
    # import torch.utils.data
    from edgeyolo import NoPrint
    

    logger.info("loading dataset...")
    dataset_cfg = params.get("dataset_cfg")
    with NoPrint():
        valdataset = get_dataset(
            cfg=dataset_cfg,
            img_size=(640, 640),
            preproc=None,
            mode="val",
        )
    return valdataset


def eval(params=None):
    import json
    dataset = load_dataset(params)
    eval_result: dict = json.load(open(params["config"], "r"))
    
    cocoFormatResult = []
    
    for i in range(len(dataset)):
        if isinstance(dataset, COCODataset):
            cur_id = dataset.ids[i]
            _, _, _, file_name, _ = dataset.annotations[i]
        else:
            cur_id = i
            file_name = dataset.annotation_list[i]["image"]
            
        
        for obj in eval_result.get(osp.basename(file_name), []):
            # if obj["score"] > 0.25:
            #     print(obj)
            obj["image_id"] = cur_id
            obj["category_id"] = dataset.class_ids[int(obj["category_id"])]
            cocoFormatResult.append(obj)

    print("num dets:", len(cocoFormatResult))
    cocoGt = dataset.coco
    cocoDt = cocoGt.loadRes(cocoFormatResult)
    
    annType = ["segm", "bbox", "keypoints"]
    
    from pycocotools.cocoeval import COCOeval
    import io
    import contextlib
    
    cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        cocoEval.summarize()
    info = redirect_string.getvalue()
    # print(cocoEval.stats)
    print(info)
    

if __name__ == '__main__':
    args = get_args()
    eval(generate_params(
        dataset_cfg=args.dataset,
        config=args.config
    ))
