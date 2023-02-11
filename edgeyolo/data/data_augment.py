import math
import random

import cv2
import numpy as np

from ..utils import xyxy2cxcywh

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
                          or single float values. Got {}".format(
                value
            )
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale, segms=None):
    import time

    def trans_points(points):       # apply affine transform
        points_c = np.ones([len(points), 3])
        points_c[..., :2] = points
        return points_c @ M.T

    twidth, theight = target_size
    # print(targets)
    if segms is None:
        num_gts = len(targets)

        # warp corner points

        corner_points = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(4 * num_gts, 2)
        corner_points = trans_points(corner_points)

        # print(corner_points)

        corner_points = corner_points.reshape(num_gts, 8)
        # print(corner_points)
        # time.sleep(1000)

        # create new boxes
        corner_xs = corner_points[:, 0::2]
        corner_ys = corner_points[:, 1::2]
        new_bboxes = (
            np.concatenate(
                (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
            )
            .reshape(4, num_gts)
            .T
        )

        # clip boxes
        new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
        new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

        targets[:, :4] = new_bboxes

    else:
        max_hw = max(twidth, theight)
        segms = [[trans_points(edge) for edge in obj] for obj in segms]
        segms = [[np.array([[min(max(x, 0), twidth) / max_hw,
                             min(max(y, 0), theight) / max_hw]
                            for x, y in edge])
                  for edge in obj]
                 for obj in segms]


        for i, obj in enumerate(segms):
            obj_points = []
            for edge in obj:
                obj_points += edge.tolist()
            obj_points = np.array(obj_points)
            # print(obj_points)
            x_min = obj_points[:, 0].min() * twidth
            x_max = obj_points[:, 0].max() * twidth
            y_min = obj_points[:, 1].min() * theight
            y_max = obj_points[:, 1].max() * theight
            try:
                targets[i, 0:4] = np.array([x_min, y_min, x_max, y_max])
            except:
                print(i, x_min, y_min, x_max, y_max)
                raise

    return targets, segms


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
    segms=None
):

    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets, segms = apply_affine_to_bboxes(targets, target_size, M, scale, segms)

    return img, targets, segms


def _mirror(image, boxes, prob=0.5, segmentations=None):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        if segmentations is not None:
            segmentations = [[np.array([1. - edge[:, 0], edge[:, 1]]).transpose()
                              for edge in obj]
                             for obj in segmentations]
    return image, boxes, segmentations


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class RandomHSV:

    def __init__(self, h=0.5, s=0.5, v=0.5) -> None:
        self.hgain = h
        self.sgain = s
        self.vgain = v

    def __call__(self, img):
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


class BaseTransform:

    max_labels = 100

    def set_max_labels(self, num=100):
        self.max_labels = num

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TrainTransform(BaseTransform):
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, hsv_gain=(0.5, 0.5, 0.5)):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.augment_hsv = RandomHSV(*hsv_gain)

    def __call__(self, image, targets, input_dim, segmentations=None):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets, None

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            self.augment_hsv(image)
        image_t, boxes, segmentations = _mirror(image, boxes, self.flip_prob, segmentations=segmentations)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        segm_reserve = None
        if segmentations is not None:
            segm_reserve = []
            [segm_reserve.append(segm) if flag else None for flag, segm in zip(mask_b, segmentations)]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels, segm_reserve


class ValTransform(BaseTransform):
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy
        self.max_labels = 100

    # assume input is cv2 img for now
    def __call__(self, img, target, input_size):
        img, r = preproc(img, input_size, self.swap)
        # target[:, :4] *= r

        # from utils

        if self.legacy:
            # img = img[::-1, :, :].copy()
            img /= 255.0
            # img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            # img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        # print(img.shape)
        return img, np.zeros((1, 5))
