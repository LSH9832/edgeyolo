import os
import os.path as osp
import cv2
import yaml
from datetime import timedelta
from loguru import logger
from platform import system

import torch
import torch.multiprocessing as mp
from torch import distributed as dist
import torch.backends.cudnn as cudnn

from .trainer import Trainer
from ..utils import init_logger

with open(os.path.join(os.path.dirname(__file__), "default.yaml")) as dpf:
    DEFAULT_PARAMS: dict = yaml.load(dpf, yaml.Loader)


def load_train_settings(train_setting_file: str):
    if train_setting_file.lower() == "default":
        return DEFAULT_PARAMS

    if osp.isfile(train_setting_file):
        with open(train_setting_file) as f:
            params = yaml.load(f, yaml.Loader)
        for k, v in DEFAULT_PARAMS.items():
            if k not in params:
                params[k] = v
                logger.info(f"param '{k}' not list in settings file, use default value: {k}={v}")
    else:
        logger.info(f"train settings file {train_setting_file} not found, use default settings.")
        params = DEFAULT_PARAMS
    return params


def train_single(
    rank=0,
    params: dict = None,
    dist_url="tcp://127.0.0.1:12345"
):
    assert params is not None, "lack of params!"
    torch.set_num_threads(params["num_threads"])
    cv2.setNumThreads(params["num_threads"])

    init_logger(os.path.join(params["output_dir"], params["log_file"]))
    params["device"] = params["device"] if isinstance(params["device"], list) else [params["device"]]
    device = params["device"][rank]
    torch.cuda.set_device(device)

    world_size = len(params["device"])
    if world_size > 1:
        try:
            dist.init_process_group(
                backend="gloo" if system() == "Linux" else "gloo",
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

    trainer = Trainer(params, rank)
    if params["eval_only"]:
        trainer.evaluate_only()
    else:
        trainer.train()


def launch(train_settings_file):
    assert os.path.isfile(train_settings_file), f"settings file {train_settings_file} does not exist!"
    assert torch.cuda.is_available(), "cuda not available, please double check your device status!"

    params = load_train_settings(train_settings_file)
    init_logger()
    if isinstance(params["device"], int):
        params["device"] = [params["device"]]
    params["device"] = [*set(params["device"])]
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
            train_single,
            nprocs=world_size,
            args=(
                params,
                dist_url
            ),
            daemon=False,
            start_method=start_method,
        )
    else:
        train_single(params=params)
