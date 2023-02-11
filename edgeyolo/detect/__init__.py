from .detector import Detector
from .export_detector import TRTDetector
from ..utils import get_color
import cv2
import numpy as np


def draw(imgs, results, class_names, line_thickness=3, draw_label=True):
    single = False
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
        single = True
    out_imgs = []
    tf = max(line_thickness - 1, 1)
    for img, result in zip(imgs, results):
        # print(img.shape)
        if result is not None:
            # print(result.shape)
            for *xywh, obj, conf, cls in result:
                c1 = (int(xywh[0]), int(xywh[1]))
                c2 = (int(xywh[2]), int(xywh[3]))
                color = get_color(int(cls), True)
                cv2.rectangle(img, c1, c2, color, line_thickness, cv2.LINE_AA)
                if draw_label:
                    label = f'{class_names[int(cls)]} {obj * conf:.2f}'
                    t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # print(img.shape)
        out_imgs.append(img)
    return out_imgs[0] if single else out_imgs
