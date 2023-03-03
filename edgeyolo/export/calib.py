import torch
from torch2trt import Dataset

import os
import cv2
import random
import numpy as np
from glob import glob
from loguru import logger

from ..data.data_augment import preproc


class CalibDataset(Dataset):

    path_list = []
    _max_num = 5120   # it doesn't make sense to use too many images for calibration

    def __init__(self,
                 dataset_path,
                 num_image=500,
                 input_size=(640, 640),
                 pixel_range=255,
                 suffix="jpg",
                 batch=1):
        self.num_img = num_image
        self.input_size = input_size
        self.path_list = []
        self.batch = batch
        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]
        for this_path in dataset_path:
            if os.path.isdir(this_path):
                self.path_list.extend(glob(os.path.join(this_path, f"*.{suffix}")))
        random.shuffle(self.path_list)
        if self.num_img > 0:
            self.path_list = self.path_list[:self.num_img]
        if len(self.path_list) > self._max_num:
            logger.info(f"To many images, cut down to {self._max_num} images for calibration.")
            self.path_list = self.path_list[:self._max_num]
        batch_num = len(self.path_list) // batch
        self.path_list = self.path_list[:batch_num * batch]
        logger.info(f"used images: {len(self.path_list)}")

        self.norm = pixel_range == 1
        # print(self.norm)

    def __getitem__(self, item):
        ret = []
        for file in self.path_list[item:item + self.batch]:
            im, _ = preproc(cv2.imread(file), self.input_size)
            if self.norm:
                im /= 255.0
            # im = np.ascontiguousarray(im, dtype=np.float16)
            ret.append(torch.from_numpy(im).unsqueeze(0).cuda())
        return [torch.cat(ret)]

    def __len__(self):
        return len(self.path_list) // self.batch

    def insert(self, item):
        self.path_list.append(item)
