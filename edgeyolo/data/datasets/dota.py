import os
from time import time

# from loguru import logger

import cv2
import numpy as np
from glob import iglob

from .datasets_wrapper import Dataset
from ..data_augment import BaseTransform


class DotaDataset(Dataset):
    """
    DOTA dataset class.
    """
    coco = None
    num_annos = 0
    idx = 0
    max_num_labels = 0
    min_ratio = 0.8

    def __init__(
        self,
        data_dir=None,
        train_dir="",
        anno_dir="",
        img_size=(640, 640),
        preproc: BaseTransform = None,
        is_train=True,
        cfg: dict = None,
        test=False,
        **kwargs
    ):
        """
        DOTA dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int or tuple): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)

        self.unused_image = 0

        if cfg is None:
            self.data_dir = data_dir
            self.train_dir = train_dir
            self.anno_dir = anno_dir

            self.class_file = kwargs.get("class_file") or os.path.join(self.data_dir, "classes.txt")
            self.names = self.load_names()
        else:
            self.data_dir = cfg.get("dataset_path")
            sets: dict = cfg.get("train" if is_train else "val" if not test else "test")
            # print(self.data_dir, sets.get("image_dir"), sets)
            self.train_dir = os.path.join(self.data_dir, sets.get("image_dir"))
            self.anno_dir = os.path.join(self.data_dir, sets.get("label"))
            self.class_file = None
            self.names = [name.replace(" ", "-") for name in cfg.get("names")]
            # print(self.anno_dir)

        self.suffix = kwargs.get("suffix") or "png"
        self.is_train = is_train
        self.img_size = np.array([*img_size])
        self.preproc = preproc

        if not is_train:
            from pycocotools.coco import COCO
            from .coco import COCOCreator
            self.coco = COCO()
            self.coco_data = COCOCreator()
            self.coco_data.change_info(data_name=os.path.basename(self.data_dir),
                                       version="0.0.0", url="", author="unknown")
            self.coco_data.add_license(name="FAKE LICENSE", url="")
            self.coco_data.load_categories(names=self.names)

        self.class_ids = [i for i in range(len(self.names))]

        print("loading DOTA dataset...")
        t = time()
        self.annotation_list = self._load_dota_annotations()

        print(f"DONE(t={time() - t:.2f}s), {len(self)} images in total.")

        if preproc is not None:
            print(f"max label number in one image: {self.max_num_labels}")
            preproc.set_max_labels(self.max_num_labels * 2)

    def __len__(self):
        return len(self.annotation_list)

    def __del__(self):
        try:
            del self.annotation_list
        except Exception as e:
            print(e)
            self.annotation_list = []

    def load_names(self):
        classes = []
        if os.path.isfile(self.class_file):
            with open(self.class_file, "r", encoding="utf8") as f:
                for name in f.read().split("\n"):
                    while name.endswith(" "):
                        name = name[:-1]
                    if len(name):
                        classes.append(name.replace(" ", "-"))
        return classes

    def get_anno_by_name(self, name):
        msg = {
            "image": os.path.join(self.train_dir, f"{os.path.basename(name)[:-4]}.{self.suffix}"),
            "annotations": []
        }
        num_labels = 0
        with open(name, "r", encoding="utf8") as f:
            has_anno = False
            for line in f.read().split("\n"):
                line = line.split()
                if len(line) == 10:
                    num_labels += 1
                    x1, x2 = sorted([float(line[0]), float(line[2])])
                    y1, y2 = sorted([float(line[1]), float(line[5])])
                    class_id = self.names.index(line[-2])
                    msg["annotations"].append([x1, y1, x2, y2, class_id])   # xywh
                    if not self.is_train:
                        if not has_anno:
                            self.coco_data.add_image(
                                image_id=self.idx - self.unused_image,
                                file_name=os.path.basename(msg["image"]),
                                width=20000,
                                height=20000,
                            )
                            has_anno = True

                        self.coco_data.add_annotation(
                            image_id=self.idx - self.unused_image,
                            anno_id=self.num_annos,
                            category_id=class_id,
                            bbox=[x1, y1, x2 - x1, y2 - y1],
                            iscrowd=0
                        )
                    self.num_annos += 1
            if not has_anno:
                self.unused_image += 1
            self.max_num_labels = max(self.max_num_labels, num_labels)
            msg["annotations"] = np.array(msg["annotations"])
            return msg

    def _load_dota_annotations(self):
        annotation_list = []
        for self.idx, anno_file in enumerate(iglob(os.path.join(self.anno_dir, "*.txt"))):
            anno_file = anno_file.replace('\\', '/')
            if os.path.isfile(os.path.join(self.train_dir, f"{os.path.basename(anno_file)[:-4]}.{self.suffix}")):
                anno = self.get_anno_by_name(anno_file)
                if not len(anno["annotations"]):
                    print(f"there are no labels in {anno_file}, skip")
                else:
                    annotation_list.append(anno)
        if not self.is_train:
            self.coco.dataset = self.coco_data.to_json()
            self.coco.createIndex()
        return annotation_list

    def save_cache(self):
        pass

    def load_anno(self, index):
        return self.annotation_list[index]["annotations"]   # [num_obj, 5(xywh + cls)]

    @staticmethod
    def knn(points, k=2, max_iter=-1):
        init_idxs = np.random.randint(0, len(points), k)
        centers = np.concatenate([[points[i]] for i in init_idxs])
        now_iter = 0
        while True:

            now_iter += 1
            groups = [[[np.array([-1, -1])]] for _ in range(k)]
            for point in points:

                try:
                    distances = np.sum(np.abs(centers - point), axis=1)
                    groups[distances.argmin()].append([point])
                except Exception as e:
                    print(centers, points.shape)
                    raise e

            try:

                new_centers = np.concatenate([[np.mean(np.concatenate(group if len(group) == 1 else group[1:]), 0)] for group in groups])
            except Exception as e:
                print(groups)
                raise e

            if (new_centers == centers).all() or now_iter == max_iter:
                # idx = np.argmax([len(group) for group in groups])
                idxs = []
                [idxs.append(i) if len(group) > 10 else None for i, group in enumerate(groups)]

                idx = idxs[np.random.randint(0, len(idxs))] if len(idxs) else np.argmax([len(group) for group in groups])
                # print(groups)
                # print(new_centers)

                return new_centers[idx]

            centers = new_centers


    def load_resized_img(self, index, res):
        img = self.load_image(index)

        r = 1.0
        crop = False
        if self.is_train:
            r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
            if r < self.min_ratio:
                r = self.min_ratio + np.random.random() * 2 * (1.0 - self.min_ratio)
                crop = True

        res[..., :4] *= r

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        # print(img.shape[:2], "->", resized_img.shape[:2], r)

        # 由于图片可能过大而目标较小，对图片进行裁剪
        if self.is_train and crop:

            # print("图像过大，进行裁剪")

            center_points = (res[..., :2] + res[..., 2:4]) * 0.5

            # 聚类得到图像中心, 进一步得到左上角的点
            x1, y1 = self.knn(center_points, k=9, max_iter=10) - self.img_size * 0.5

            x1 = max(int(x1), 0)
            y1 = max(int(y1), 0)

            # 如果以此构建的图像 右下角的点超出图像边界，则拉回到边界
            if x1 + self.img_size[1] > resized_img.shape[1]:
                x1 = resized_img.shape[1] - self.img_size[1]
            if y1 + self.img_size[0] > resized_img.shape[0]:
                y1 = resized_img.shape[0] - self.img_size[0]



            # 右下角的点
            x2 = x1 + self.img_size[1]
            y2 = y1 + self.img_size[0]


            # 如果左上角的点超出边界，说明长或宽没有达到预设长度，在该维度不需要裁剪
            if x1 < 0:
                x1 = 0
                x2 = resized_img.shape[1]

            if y1 < 0:
                y1 = 0
                y2 = resized_img.shape[0]

            # print(f"聚类左上角：({x1},{y1})")
            # print(f"聚类右下角：({x2},{y2})")

            # 剔除不在范围内的标签
            cxs = (res[..., 0] + res[..., 2]) * 0.5
            cys = (res[..., 1] + res[..., 3]) * 0.5
            mask = (x1 <= cxs) * (cxs <= x2) * (y1 <= cys) * (cys <= y2)

            res = res[mask]
            res[..., 0].clip(x1, x2)
            res[..., 1].clip(y1, y2)
            res[..., 2].clip(x1, x2)
            res[..., 3].clip(y1, y2)
            res[..., 0:2] -= np.array([x1, y1])
            res[..., 2:4] -= np.array([x1, y1])

            # print(x1, x2, y1, y2)

            resized_img = resized_img[y1:y2, x1:x2]
            # print(resized_img.shape[:2])

        return resized_img, res, (img.shape[0], img.shape[1])

    def load_image(self, index):
        img_file = self.annotation_list[int(index)]["image"].replace('\\', '/')
        img = cv2.imread(img_file)
        assert img is not None, f"File {img_file} does not exist or broken!"
        return img

    def pull_item(self, index):
        # rectangles, img_info, resized_info, filename, segments
        anno = self.annotation_list[index]
        res = anno["annotations"].copy()

        # print(index, os.path.basename(anno["image"]))
        img, res, img_info = self.load_resized_img(index, res)

        return img, res, img_info, np.array([index]), None


    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id, segments = self.pull_item(index)
        # print(img_info)
        if self.preproc is not None:
            if self.is_train:
                img, target, segments = self.preproc(img, target, self.input_dim, segments)
            else:
                img, target = self.preproc(img, target, self.input_dim)
                segments = 0

        return img, target, img_info, img_id, segments
