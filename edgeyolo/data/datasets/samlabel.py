import os
from time import time

# from loguru import logger

import cv2
import numpy as np
import pickle
import json
from glob import iglob, glob

from .datasets_wrapper import Dataset


class SAMLabelDataset(Dataset):

    coco = None
    num_annos = 0
    idx = 0
    max_num_labels = 0
    min_ratio = 0.8
    coco_data = None

    def __init__(
        self,
        img_size=(640, 640),
        preproc=None,
        is_train=True,
        cfg: dict = None,
        test=False,
        **kwargs
    ):
        """
        SAMLabel dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int or tuple): target image size after pre-processing
            preproc: data augmentation strategy
        """
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        super().__init__(img_size)

        self.unused_image = 0
        self.test = test

        self.data_dir = cfg.get("dataset_path")
        sets: dict = cfg.get("train" if is_train else "val" if not test else "test")
        self.train_dir = os.path.join(self.data_dir, sets.get("image_dir"))
        if not test:
            self.anno_dir = os.path.join(self.data_dir, sets.get("label"))
        self.class_file = None
        self.names = cfg.get("names")
        # print(self.anno_dir)

        self.suffix = kwargs.get("suffix", "jpg")
        self.use_cache = kwargs.get("use_cache", False)
        self.use_segm = cfg.get("segmentaion_enabled", False)
        self.is_train = is_train
        self.img_size = np.array([*img_size])
        
        self.preproc = preproc
        self.segm_len = 0
        

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
        self.class_id_map = {}
        for i, name in enumerate(self.names):
            self.class_id_map[name] = i

        print("loading samlabel dataset...")
        t = time()

        self.annotation_list = self._load_samlabel_annotations()

        print(f"DONE(t={time() - t:.2f}s), {len(self)} images in total.")

        if preproc is not None:
            print(f"max label number in one image: {self.max_num_labels}")
            preproc.set_max_labels(max(self.max_num_labels * 5, 50))

    def __len__(self):
        return len(self.annotation_list)

    def __del__(self):
        try:
            del self.annotation_list
        except Exception as e:
            print(e)
            self.annotation_list = []

    def get_anno_by_name(self, name):
        img_cfg: dict = json.load(open(name, "r"))
        msg = {
            "image": os.path.join(self.train_dir, img_cfg['info']['name']),
            "annotations": [],
            "segmentations": [] if self.use_segm else None
        }
        if not os.path.isfile(msg["image"]):
            return None, False

        num_labels = len(img_cfg['objects'])
        img_h = img_cfg['info']['height']
        img_w = img_cfg['info']['width']

        has_anno = False
        group_objs = {}
        max_wh = max(img_w, img_h)
        for obj in img_cfg['objects']:
            x1, y1, x2, y2 = obj['bbox']
            class_id = self.class_id_map[obj['category']]

            segm: np.ndarray = None
            if self.use_segm:
                segm = (np.asarray(obj['segmentation'], dtype=float) / max_wh).tolist()
                self.segm_len = max(len(segm), self.segm_len)

            no_group = isinstance(obj['group'], str) and not len(obj['group'])
            if not no_group:
                group_name = f"{class_id}_{obj['group']}"
                if group_objs.get(group_name, None) is not None:
                    group_objs[group_name][0][0] = min(x1/img_w, group_objs[group_name][0][0])
                    group_objs[group_name][0][1] = min(y1/img_h, group_objs[group_name][0][1])
                    group_objs[group_name][0][2] = max(x2/img_w, group_objs[group_name][0][2])
                    group_objs[group_name][0][3] = max(y2/img_h, group_objs[group_name][0][3])
                    if segm is not None:
                        group_objs[group_name][1].append(segm)
                else:
                    group_objs[group_name] = [[x1/img_w, y1/img_h, x2/img_w, y2/img_h, class_id], [segm] if segm is not None else []]
                continue

            msg["annotations"].append([x1/img_w, y1/img_h, x2/img_w, y2/img_h, class_id])   # xywh
            if self.use_segm:
                msg["segmentations"].append([segm] if segm is not None else [])
            
            if not self.is_train:
                if not has_anno:
                    self.coco_data.add_image(
                        image_id=self.idx - self.unused_image,
                        file_name=os.path.basename(msg["image"]),
                        width=img_w,
                        height=img_h,
                    )
                    has_anno = True

                self.coco_data.add_annotation(
                    image_id=self.idx - self.unused_image,
                    anno_id=self.num_annos,
                    category_id=class_id,
                    bbox=[x1, y1, x2-x1, y2-y1], # xywh
                    iscrowd=0
                )
                self.num_annos += 1
        
        for _, v in group_objs.items():
            msg["annotations"].append(v[0])
            if self.use_segm:
                msg["segmentations"].append(v[1])
            if not self.is_train:
                if not has_anno:
                    self.coco_data.add_image(
                        image_id=self.idx - self.unused_image,
                        file_name=os.path.basename(msg["image"]),
                        width=img_w,
                        height=img_h,
                    )
                    has_anno = True

                self.coco_data.add_annotation(
                    image_id=self.idx - self.unused_image,
                    anno_id=self.num_annos,
                    category_id=v[0][-1],
                    bbox=[v[0][0], v[0][1], v[0][2]-v[0][0], v[0][3]-v[0][1]], # xywh
                    iscrowd=0
                )
                self.num_annos += 1

        if not has_anno:
            self.unused_image += 1
        self.max_num_labels = max(self.max_num_labels, num_labels)
        msg["annotations"] = np.array(msg["annotations"])
        return msg, True

    def _load_samlabel_annotations(self):

        cache_file = self.cache_file = os.path.join(
            self.data_dir,
            f"{'train' if self.is_train else 'val' if not self.test else 'test'}_"
            f"cache_samlabel_{self.img_size[0]}x{self.img_size[1]}.edgeyolo"
        ).replace("\\", "/")
        self.cached = True
        if os.path.isfile(cache_file) and self.use_cache:
            print(f"loading cache from {cache_file}.")

            with open(cache_file, "rb") as f:
                annotation_list, self.coco_data, self.max_num_labels, self.segm_len = pickle.load(f)
        else:
            self.cached = False
            annotation_list = []
            file_list = glob(os.path.join(self.anno_dir, "*.json"))
            num_file = len(file_list)
            for self.idx, anno_file in enumerate(file_list):

                print(f"\rprocessing labels: {self.idx + 1} / {num_file}", end="")

                anno_file = anno_file.replace('\\', '/')
                
                anno, success = self.get_anno_by_name(anno_file)
                if success:
                    if not len(anno["annotations"]):
                        print(f"\nthere are no labels in {anno_file}, skip")
                    else:
                        annotation_list.append(anno)
            print()
            # with open(cache_file, "wb") as cachef:
            #     pickle.dump((annotation_list, self.coco_data, self.max_num_labels), cachef)


        if not self.is_train:
            self.coco_data.show_each_category_num()
            self.coco.dataset = self.coco_data.to_json()
            self.coco.createIndex()
        return annotation_list

    def save_cache(self):
        if self.cached:
            return
        with open(self.cache_file, "wb") as cachef:
            pickle.dump((self.annotation_list, self.coco_data, self.max_num_labels, self.segm_len), cachef)

    def load_anno(self, index):
        return self.annotation_list[index]["annotations"]   # [num_obj, 5(xywh + cls)]

    def load_resized_img(self, index, res):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])

        img_h, img_w = img.shape[:2]

        # print(res)

        res[..., 0] *= img_w
        res[..., 2] *= img_w
        res[..., 1] *= img_h
        res[..., 3] *= img_h

        res[..., :4] *= r

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img, res, (img.shape[0], img.shape[1])

    def load_image(self, index, get_file_name=False):
        img_file = self.annotation_list[index]["image"].replace('\\', '/')
        img = cv2.imread(img_file)
        assert img is not None, f"Can not read image from {img_file}!"
        if get_file_name:
            return img, img_file
        return img

    def pull_origin_item(self, index):
        image, fn = self.load_image(index, True)
        res = self.annotation_list[index]["annotations"].copy()
        img_h, img_w = image.shape[:2]
        res[..., [0, 2]] *= img_w
        res[..., [1, 3]] *= img_h
        return image, res, fn

    def pull_item(self, index):
        """
        Returns:
          resized_img, rectangles, origin_img_size, idx, segments
        """
        # print(len(self.annotation_list))
        anno = self.annotation_list[index]
        res = anno["annotations"].copy()
        img, res, img_info = self.load_resized_img(index, res)
        return img, res, img_info, np.array([index]), anno["segmentations"].copy()

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
