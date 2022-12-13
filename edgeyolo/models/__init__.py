from .yolo import *
# import yaml
import torch
import os
from loguru import logger


def load_model(cfg_file) -> Model:

    def init_weight(M):
        from torch import nn
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    my_model = Model(cfg_file, is_file=os.path.isfile(cfg_file))
    my_model.apply(init_weight)
    return my_model


class EdgeYOLO:

    model = None
    cfg_data = None
    optimizer = None
    dataloader = None
    evaluator = None
    lr_scheduler = None
    loss = None
    start_epoch = 0
    now_epoch = 0
    input_size = [640, 640]
    class_names = []
    rank = 0
    is_match = True

    def __init__(self, cfg_file=None, weights=None, rank=0, write_cfg_to_weights=False):
        self.rank = rank
        assert cfg_file is not None or weights is not None

        self.__weights = weights
        if weights is not None and os.path.isfile(weights):
            if not rank:
                logger.info(f"loading models from weight {os.path.abspath(weights)}")
            self.ckpt = torch.load(weights, map_location="cpu")
            if write_cfg_to_weights:
                if cfg_file is not None:
                    self.ckpt["cfg_data"] = open(cfg_file, "r", encoding="utf8").read()
                    if not rank:
                        logger.info(f"use cfg data {cfg_file}")
                        f = os.path.join(os.path.dirname(weights),
                                         os.path.basename(weights).split(".")[0] + "(cfg_change).pth")
                        torch.save(self.ckpt, f)

            self.cfg_data = self.ckpt["cfg_data"]
            # print(self.cfg_data)
            self.model = load_model(self.cfg_data)
            try:
                self.model.load_state_dict(self.ckpt["model"], strict=False)
            except:
                self.is_match = self.try_load_state_dict(self.ckpt["model"])
            if "epoch" in self.ckpt:
                self.start_epoch = self.ckpt["epoch"] + 1
                self.now_epoch = max(0, self.start_epoch - 1)
            if "class_names" in self.ckpt:
                self.class_names = self.ckpt["class_names"]
                # print(self.ckpt["class_names"])
        elif cfg_file is not None and os.path.isfile(cfg_file):
            if not rank:
                logger.info(f"no weight file found, setup models from cfg file {os.path.abspath(cfg_file)}")

            self.model = load_model(cfg_file)
            self.cfg_data = open(cfg_file).read()
        else:
            assert False

    def load_class_names(self, class_names_file):
        self.class_names = []
        [self.class_names.append(name) if len(name) and name not in self.class_names else None
         for name in open(class_names_file).read().split("\n")]
        # print(self.class_names)
        assert len(self.class_names) == len(self.model.names), f"{len(self.class_names), len(self.model.names)}"
        self.ckpt["class_names"] = self.class_names

    def resave(self):
        torch.save(self.ckpt, self.__weights)
        logger.info("weights updated")

    def save(self, filename, data=None):
        if isinstance(filename, str):
            filename = [filename]
        save_data = {
            "cfg_data": self.cfg_data,
            "epoch": self.now_epoch,
            "class_names": self.class_names
        }
        if self.model is not None:
            save_data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            save_data["optimizer"] = self.optimizer.state_dict()

        if data is not None:
            for k, v in data.items():
                save_data[k] = v

        for fn in filename:
            torch.save(save_data, fn)
            if not self.rank:
                logger.info(f"weight file saved to {fn}")

    def try_load_state_dict(self, state_dict, distributed=False):
        is_match = True
        if isinstance(state_dict, str) and os.path.isfile(state_dict):
            state_dict = torch.load(state_dict, map_location="cpu")["model"]

        if distributed:
            try:
                self.model = self.model.module
            except:
                pass
        for k, v in state_dict.items():
            try:
                self.model.load_state_dict({k: v}, strict=False)
            except RuntimeError as e:
                err = str(e).split("\n")[-1]
                while err.startswith(" ") or err.startswith("\t"):
                    err = err[1:]
                if not self.rank:
                    logger.warning(err)
                    is_match = False
        return is_match

    def get_model_size(self, input_size=None):
        from ..utils import get_model_info
        return get_model_info(self.model, self.input_size if input_size is None else input_size)
