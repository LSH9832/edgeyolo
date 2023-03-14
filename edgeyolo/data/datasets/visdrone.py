import os
from time import time

# from loguru import logger

import cv2
import numpy as np
import pickle
from glob import iglob

from .datasets_wrapper import Dataset
from ..data_augment import BaseTransform


class VisDroneDataset(Dataset):
    """
    VisDrone dataset class.
    """
    coco = None
    num_annos = 0
    idx = 0
    max_num_labels = 0
    min_ratio = 0.8
    coco_data = None

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

        self.test = test
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
            if not test:
                self.anno_dir = os.path.join(self.data_dir, sets.get("label"))
            self.class_file = None
            self.names = cfg.get("names")
            # print(self.anno_dir)

        self.suffix = kwargs.get("suffix") or "jpg"
        self.use_cache = kwargs.get("use_cache") or False
        self.is_train = is_train
        self.img_size = np.array([*img_size])
        self.preproc = preproc

        if not is_train and not test:
            from pycocotools.coco import COCO
            from .coco import COCOCreator
            self.coco = COCO()
            self.coco_data = COCOCreator()
            self.coco_data.change_info(data_name=os.path.basename(self.data_dir),
                                       version="0.0.0", url="", author="unknown")
            self.coco_data.add_license(name="FAKE LICENSE", url="")
            self.coco_data.load_categories(names=self.names)

        self.class_ids = [i for i in range(len(self.names))]

        print("loading VisDrone dataset...")
        t = time()
        self.annotation_list = self._load_visdrone_annotations()

        print(f"DONE(t={time() - t:.2f}s), {len(self)} images in total.")

        if preproc is not None:
            print(f"max label number in one image: {self.max_num_labels}")
            preproc.set_max_labels(max(self.max_num_labels * 2, 50))

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

                line = line.replace(",", " ").split()
                if len(line) == 8:
                    x1, y1, w, h = [float(k) for k in line[:4]]
                    x2, y2 = x1 + w, y1 + h
                    score = int(line[4])
                    if score == 0:
                        continue

                    class_id = int(line[5])
                    if class_id == 0 or class_id > len(self.names):
                        # id=0: ignore regions
                        continue
                    class_id -= 1
                    num_labels += 1
                    msg["annotations"].append([x1, y1, x2, y2, class_id])   # xywh
                    if not self.is_train:
                        if not has_anno:
                            self.coco_data.add_image(
                                image_id=self.idx - self.unused_image,
                                file_name=os.path.basename(msg["image"]),
                                width=4000,
                                height=4000,
                            )
                            has_anno = True

                        self.coco_data.add_annotation(
                            image_id=self.idx - self.unused_image,
                            anno_id=self.num_annos,
                            category_id=class_id,
                            bbox=[x1, y1, w, h],
                            iscrowd=0
                        )
                    self.num_annos += 1
            if not has_anno:
                self.unused_image += 1
            self.max_num_labels = max(self.max_num_labels, num_labels)
            msg["annotations"] = np.array(msg["annotations"])
            return msg

    def _load_visdrone_annotations(self):

        cache_file = self.cache_file = os.path.join(
            self.data_dir,
            f"{'train' if self.is_train else 'val' if not self.test else 'test'}_"
            f"cache.edgeyolo"
        ).replace("\\", "/")
        self.cached = True
        if os.path.isfile(cache_file) and self.use_cache:
            print(f"loading cache from {cache_file}.")

            with open(cache_file, "rb") as f:
                annotation_list, self.coco_data, self.max_num_labels = pickle.load(f)
        else:
            self.cached = False
            annotation_list = []
            for self.idx, anno_file in enumerate(iglob(os.path.join(self.anno_dir, "*.txt"))):
                anno_file = anno_file.replace('\\', '/')
                if os.path.isfile(os.path.join(self.train_dir, f"{os.path.basename(anno_file)[:-4]}.{self.suffix}")):
                    anno = self.get_anno_by_name(anno_file)
                    if not len(anno["annotations"]):
                        print(f"there are no labels in {anno_file}, skip")
                    else:
                        annotation_list.append(anno)

            # with open(cache_file, "wb") as cachef:
            #     pickle.dump((annotation_list, self.coco_data, self.max_num_labels), cachef)

        if not self.is_train:
            self.coco.dataset = self.coco_data.to_json()
            self.coco.createIndex()
        return annotation_list

    def save_cache(self):
        if self.cached:
            return
        with open(self.cache_file, "wb") as cachef:
            pickle.dump((self.annotation_list, self.coco_data, self.max_num_labels), cachef)

    def load_anno(self, index):
        return self.annotation_list[index]["annotations"]   # [num_obj, 5(xywh + cls)]

    def load_resized_img(self, index, res):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        res[..., :4] *= r

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img, res, (img.shape[0], img.shape[1])

    def load_image(self, index):
        img_file = self.annotation_list[index]["image"].replace('\\', '/')
        img = cv2.imread(img_file)
        assert img is not None, f"File {img_file} does not exist or broken!"
        return img

    def pull_item(self, index):
        """
        Returns:
          resized_img, rectangles, origin_img_size, idx, segments
        """
        anno = self.annotation_list[index]
        res = anno["annotations"].copy()

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
