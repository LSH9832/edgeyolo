# import numpy as np
# import cv2
import torch.nn as nn


def preprocess(inputs, targets, input_size):
    _, _, h, w = inputs.shape
    scale_y = input_size[0] / h
    scale_x = input_size[1] / w
    if scale_x != 1 or scale_y != 1:
        inputs = nn.functional.interpolate(
            inputs, size=input_size, mode="bilinear", align_corners=False
        )
        targets[..., 1::2] = targets[..., 1::2] * scale_x
        targets[..., 2::2] = targets[..., 2::2] * scale_y
    return inputs, targets
