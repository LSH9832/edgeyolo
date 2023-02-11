from edgeyolo.data import MosaicDetection, TrainTransform, get_dataset
from edgeyolo.detect import get_color
import cv2
import numpy as np


def load():
    img_size = (1280, 1280)

    import time

    t0 = time.time()

    dataset = get_dataset(
        cfg="../params/dataset/coco.yaml",
        img_size=img_size,
        preproc=TrainTransform(
            max_labels=500,
            flip_prob=0.5,
            hsv_prob=1,
            hsv_gain=[0.0138, 0.664, 0.464]
        ),
        mode="train"
    )

    print(F"TOTAL TIME COST ON LOADING DATASET: {time.time() - t0}s")

    dataset = MosaicDetection(
        dataset,
        mosaic=True,
        img_size=img_size,
        preproc=dataset.preproc,
        degrees=float(10),
        translate=0.1,
        mosaic_scale=(0.1, 2),
        mixup_scale=(0.5, 1.5),
        shear=2.0,
        enable_mixup=True,
        mosaic_prob=0,
        mixup_prob=0,
        rank=0,
        train_mask=False
    )

    print(len(dataset))
    for i in range(len(dataset)):

        # i=979
        mix_img, padded_labels, *_ = dataset[i]
        # print(padded_labels)

        mix_img = np.ascontiguousarray(mix_img.transpose((1, 2, 0)), dtype="uint8")
        # print(dataset._dataset.annotation_list[i]["image"], mix_img.shape)

        for cls, *xywh in padded_labels:
            if not sum(xywh) == 0:
                cx, cy, w, h = xywh
                cv2.rectangle(mix_img,
                              (int(cx - w / 2), int(cy - h / 2)),
                              (int(cx + w / 2), int(cy + h / 2)),
                              get_color(cls, bgr=True), 1, cv2.LINE_AA)
        print(f"{i+1}/{len(dataset)}")
        cv2.imshow("dataset", mix_img)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    load()
