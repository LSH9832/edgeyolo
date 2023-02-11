import random

import cv2
import numpy as np

from ...utils import adjust_box_anns

from ..data_augment import random_affine
from .datasets_wrapper import Dataset
from .coco import COCODataset
from .mask_coding import *


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w, mixup=False):
    # TODO update doc
    # index0 to top left part of image
    if mixup:
        x1, y1, x2, y2 = int(max(xc - w / 2, 0)), \
                         int(max(yc - h / 2, 0)), \
                         int(min(xc + w / 2, input_w * 2)), \
                         int(min(input_h * 2, yc + h / 2))
        small_coord = int((w - (x2 - x1)) / 2), int((h - (y2 - y1)) / 2), int((w + (x2 - x1)) / 2), int((h + (y2 - y1)) / 2)
        # print((x1, y1, x2, y2), small_coord)
    else:
        if mosaic_index == 0:
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            small_coord = w - (x2 - x1), h - (y2 - y1), w, h
        # index1 to top right part of image
        elif mosaic_index == 1:
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
            small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        # index2 to bottom left part of image
        elif mosaic_index == 2:
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
            small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        # index2 to bottom right part of image
        elif mosaic_index == 3:
            x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
            small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)

    return (x1, y1, x2, y2), small_coord


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, rank=0
    ):
        """

        Args:
            dataset(COCODataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            enable_mixup (bool):
            rank (int):
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = rank

    def __len__(self):
        return len(self._dataset)

    def __get_mosaic_item(self, idx, mixup=False):
        while True:
            mosaic_labels = []
            segments = []

            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # 横纵坐标中点
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices， 加上其他三张图片
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3 if mixup else 0)]

            this_img_id = 0
            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id, segms = self._dataset.pull_item(index)
                while not len(_labels):
                    index = random.randint(0, len(self._dataset) - 1)
                    img, _labels, _, img_id, segms = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                mixup_scale = 2 if mixup else 1
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                    this_img_id = img_id

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w, mixup
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)
                if segms is not None:
                    segms = [[np.array([[min(max(x * input_w + padw, 0), 2 * input_w),
                                         min(max(y * input_h + padh, 0), 2 * input_h)]
                                        for x, y in edge])
                              for edge in obj]
                             for obj in segms]

                    segments += segms
                else:
                    segments = None

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            # print("random affine")

            mosaic_img, mosaic_labels, segments = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale if not mixup else self.mixup_scale,
                shear=self.shear,
                segms=segments
            )
            try:
                mix_img, padded_labels, segments = self.preproc(mosaic_img, mosaic_labels, self.input_dim, segments)
            except:
                print("somthing went wrong", mosaic_labels, segments, idx)
                cv2.waitKey(0)
                raise
                # continue
            zero_mask = padded_labels[..., 3] * padded_labels[..., 4] == 0
            padded_labels[zero_mask] *= 0
            if segments is None or len(segments) > 0:
                break
        return mix_img, padded_labels, segments, this_img_id

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_img, padded_labels, segments, img_id = self.__get_mosaic_item(idx)
            final_len = self.preproc.max_labels
            if (
                self.enable_mixup
                and not len(padded_labels) == 0
                and random.random() < self.mixup_prob
            ):
                # mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
                add_idx = random.randint(0, len(self._dataset) - 1)
                add_img, add_labels, add_segments, _ = self.__get_mosaic_item(add_idx, True)
                mosaic_img = mosaic_img // 2 + add_img // 2
                if segments is None:
                    padded_labels = np.vstack((padded_labels, add_labels))
                    final_len = self.preproc.max_labels * 2
                else:
                    padded_labels = np.vstack((padded_labels[:len(segments)], add_labels))
                    segments += add_segments

            # mix_img, padded_labels, segments = self.preproc(mosaic_img, mosaic_labels, self.input_dim, segments)
            img_info = (mosaic_img.shape[1], mosaic_img.shape[0])

            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            if segments is None:
                segments = [-1]
            else:
                segments = encode_mask(segments, max_obj_num=final_len)
            return mosaic_img, padded_labels[:final_len], img_info, img_id, segments

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id, segms = self._dataset.pull_item(idx)
            while not len(label):
                img, label, img_info, img_id, segms = self._dataset.pull_item(random.randint(0, len(self._dataset) - 1))
            img, label, segms = self.preproc(img, label, self.input_dim, segms)
            if segms is None:
                segms = [-1]
            else:
                segms = encode_mask(segms, max_obj_num=len(label))
            return img, label, img_info, img_id, segms

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _, segms = self._dataset.pull_item(cp_index)
        # print(len(segms))

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels
