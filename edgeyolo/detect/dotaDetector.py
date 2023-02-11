from ..models import EdgeYOLO
from ..data.data_augment import preproc
from ..utils import postprocess, get_model_info

import torch
import numpy as np
from time import time
import cv2
import torchvision


class DOTADetector:
    """
    Single image inference only
    """

    infer_time = 0.
    nms_time = 0.

    def __init__(self, model=None, conf=0.01, nms=0.65, nc=18, obj_conf=True, half=False, img_size=960, normal=False,
                 twice_nms=False):
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.model.half() if half else None
        self.conf = conf
        self.nms = nms
        self.nc = nc
        self.obj_conf = obj_conf
        self.half = half
        self.win_size = ([img_size] * 2) if isinstance(img_size, int) else img_size
        self.block_size = self.win_size[0] // 3
        self.min_img_size = [self.block_size * 4] * 2
        self.normal = normal
        self.twice_nms = twice_nms

    def normal_inference(self, img, class_agnostic=False):
        with torch.no_grad():
            r = min(self.win_size[0] / img.shape[0], self.win_size[1] / img.shape[1])
            img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))


            pad_img = (np.ones(self.win_size + [3]) * 114).astype("uint8")
            pad_img[:img.shape[0], :img.shape[1]] = img[:, :]

            ipt = torch.from_numpy(pad_img.transpose(2, 0, 1)).unsqueeze(0)
            ipt = ipt.cuda().float()
            if self.half:
                ipt = ipt.half()
            t0 = time()
            result = self.model(ipt)
            t1 = time()
            self.infer_time += t1 - t0

            # [x1, y1, x2, y2, conf0, conf1, cls]

            result = postprocess(
                result.cpu(),
                num_classes=self.nc,
                conf_thre=self.conf,
                nms_thre=self.nms,
                class_agnostic=class_agnostic,
                obj_conf_enabled=self.obj_conf
            )[0]
            if result is not None:
                result[..., :4] /= r
            return result


    def __call__(self, img: np.ndarray, class_agnostic=False):

        if self.normal:
            return self.normal_inference(img, class_agnostic)

        with torch.no_grad():
            self.infer_time = 0
            self.nms_time = 0

            r = 1.0
            ratio = min(self.min_img_size[0] / img.shape[0], self.min_img_size[1] / img.shape[1])
            if ratio > 1:
                r = ratio
                img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))


            pad_img_blocks = [img.shape[0] // self.block_size + (0 if img.shape[0] % self.block_size == 0 else 1),
                              img.shape[1] // self.block_size + (0 if img.shape[1] % self.block_size == 0 else 1)]
            pad_img_size = [s * self.block_size for s in pad_img_blocks] + [3]

            pad_img = (np.ones(pad_img_size, dtype="uint8") * 114).astype("uint8")
            try:
                pad_img[:img.shape[0], :img.shape[1]] = img[:, :]
            except:
                from loguru import logger
                logger.info(f"{pad_img.shape}, {img.shape}")
                raise

            results = []
            for j in range(pad_img_blocks[0] - 2):
                for i in range(pad_img_blocks[1] - 2):
                    x1, y1 = i * self.block_size, j * self.block_size
                    x2, y2 = x1 + self.win_size[1], y1 + self.win_size[0]

                    now_window = pad_img[y1:y2, x1:x2]
                    ipt = torch.from_numpy(now_window.transpose(2, 0, 1)).unsqueeze(0)
                    ipt = ipt.cuda().float()
                    if self.half:
                        ipt = ipt.half()

                    t0 = time()
                    result = self.model(ipt)
                    t1 = time()
                    self.infer_time += t1 - t0

                    # [x1, y1, x2, y2, conf0, conf1, cls]


                    result = postprocess(
                        result.cpu(),
                        num_classes=self.nc,
                        conf_thre=self.conf,
                        nms_thre=self.nms,
                        class_agnostic=class_agnostic,
                        obj_conf_enabled=self.obj_conf
                    )[0]
                    t2 = time()
                    self.nms_time += t2 - t1

                    if result is not None:
                        if not self.twice_nms:
                            cxs = (result[:, 0] + result[:, 2]) * 0.5
                            cys = (result[:, 1] + result[:, 3]) * 0.5

                            use_blocks = [[1, 1]]
                            if i == j == 0:
                                use_blocks.append([0, 0])
                            if i == 0:
                                use_blocks.append([0, 1])
                            if j == 0:
                                use_blocks.append([1, 0])
                            if i == pad_img_blocks[1] - 3 and j == pad_img_blocks[0] - 3:
                                use_blocks.append([2, 2])
                            if i == pad_img_blocks[1] - 3:
                                use_blocks.append([2, 1])
                            if j == pad_img_blocks[0] - 3:
                                use_blocks.append([1, 2])
                            if i == 0 and j == pad_img_blocks[0] - 3:
                                use_blocks.append([0, 2])
                            if i == pad_img_blocks[1] - 3 and j == 0:
                                use_blocks.append([2, 0])

                            mask = None
                            for block_i, block_j in use_blocks:
                                bx1, by1 = block_i * self.block_size, block_j * self.block_size
                                bx2, by2 = bx1 + self.block_size, by1 + self.block_size

                                block_mask = (bx1 <= cxs) * (cxs < bx2) * (by1 <= cys) * (cys < by2)
                                if mask is None:
                                    mask = block_mask
                                else:
                                    mask += block_mask

                            result = result[mask]


                        if len(result):

                            result[:, 0] += x1
                            result[:, 1] += y1
                            result[:, 2] += x1
                            result[:, 3] += y1

                            results.append(result)

            torch.cuda.empty_cache()

            if len(results):
                results = torch.cat(results)

                if self.twice_nms:
                    if class_agnostic:
                        nms_out_index = torchvision.ops.nms(
                            results[:, :4],
                            results[:, 4] * results[:, 5],
                            self.nms,
                        )
                    else:
                        nms_out_index = torchvision.ops.batched_nms(
                            results[:, :4],
                            results[:, 4] * results[:, 5],
                            results[:, 6],
                            self.nms,
                        )

                    results = results[nms_out_index]

                results[..., 0:4] /= r
                return results
            return None

    @staticmethod
    def result2coco_json(result, img_id):
        results = []
        if result is not None:
            result[:, 2] -= result[:, 0]
            result[:, 3] -= result[:, 1]

            for x, y, w, h, c0, c1, cls in result:

                results.append({
                    "image_id": int(img_id),
                    "category_id": int(cls),
                    "bbox": [x, y, w, h],
                    "score": c0 * c1,
                    "segmentation": [],
                })

        return results
