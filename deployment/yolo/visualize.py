import cv2
import numpy as np


class __Colors:
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


get_color = __Colors()


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
            if result.shape[1] == 6:
                for *xywh, cls, conf in result:
                    c1 = (int(xywh[0]), int(xywh[1]))
                    c2 = (int(xywh[2]), int(xywh[3]))
                    color = get_color(int(cls), True)
                    cv2.rectangle(img, c1, c2, color, line_thickness, cv2.LINE_AA)
                    if draw_label:
                        label = f'{conf:.2f} {class_names[int(cls)]}'
                        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            elif result.shape[1] == 7:
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