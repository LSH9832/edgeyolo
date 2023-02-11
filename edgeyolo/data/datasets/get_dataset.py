import yaml
from .voc import VOCDataset
from .coco import COCODataset
from .dota import DotaDataset
from .visdrone import VisDroneDataset
import os.path as osp


datasets = {
    "voc": VOCDataset,
    "coco": COCODataset,
    "dota": DotaDataset,
    "visdrone": VisDroneDataset
}


def get_dataset(cfg, img_size=(640, 640), preproc=None, mode="train"):


    modes = {
        "train": {"is_train": True, "test": False},
        "val": {"is_train": False, "test": False},
        "test": {"is_train": False, "test": True}
    }
    if mode not in modes:
        mode = "train"

    if isinstance(cfg, str):
        if osp.isfile(cfg):
            with open(cfg, "r") as f:
                cfg = yaml.load(f, yaml.SafeLoader)

    assert isinstance(cfg, dict)

    dataset = datasets.get(cfg.get("type").lower() or "coco") or COCODataset
    return dataset(
        cfg=cfg,
        img_size=img_size,
        preproc=preproc,
        **modes.get(mode),
        **cfg["kwargs"]
    )
