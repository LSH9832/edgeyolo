import os
from time import time

# from loguru import logger

import cv2
import numpy as np
from glob import iglob
import pickle

from .datasets_wrapper import Dataset
from ..data_augment import BaseTransform


class XML:
    string = ""

    def __init__(self, string: str or list):
        self.string = string
        self.is_str = isinstance(self.string, str)
        self.iter_idx = 0

    def __repr__(self):
        return self.string

    def __getitem__(self, item):
        return self.string[item]

    def __iter__(self):
        yield from self.string

    def to_list(self):
        return [s if isinstance(s, XML) else XML(s) for s in self.string]

    def to_string(self):
        return self.string if self.is_str else [s.to_string() for s in self.string]

    def to_int(self):
        return int(self.string) if self.is_str else [s.to_int() for s in self.string]

    def to_bool(self):

        k = {
            'true': True,
            'false': False,
            'yes': True,
            'no': False,
            'y': True,
            'n': False
        }

        def _to_bool(s: str):
            return bool(int(s)) if s.isdigit() else k.get(s.lower())

        return _to_bool(self.string) if self.is_str else [s.to_bool() for s in self.string]

    def to_float(self):
        return float(self.string) if self.is_str else [s.to_float() for s in self.string]

    def find(self, patten, idx=-1):
        """
        get string in keyword patten
        :param patten: string
        :param idx: index need
        :return: string in between (XML)
        """
        results = []
        if isinstance(patten, list):
            for p in patten:
                results.append(self.find(p, idx))

        elif isinstance(patten, str):
            start = "<%s>" % patten
            end = "</%s>" % patten
            now_index = 0
            while now_index < len(self.string) - len(start) - len(end):
                if self.string[now_index:].startswith(start):
                    flag = 1
                    n_i = now_index + len(start)
                    while n_i < len(self.string) - len(start):
                        if self.string[n_i:].startswith(start):
                            flag += 1
                        elif self.string[n_i:].startswith(end):
                            flag -= 1
                            if flag == 0:
                                # print(now_index + len(start), n_i-1)
                                results.append(XML(string=self.string[now_index + len(start):n_i]))
                                if len(results) == idx + 1:
                                    return results[-1]
                                break
                        n_i += 1
                now_index += 1
        else:
            raise TypeError("Input patten must be list[XML] or str!")

        return XML(results)


def decode_VOC(fp):
    """
    get key infomation of annotation of VOC format
    :param fp: xml file
    :return: information data (dict)
    """
    if isinstance(fp, str):
        assert os.path.isfile(fp)
        fp = open(fp)

    my_xml = XML(fp.read())

    objects = my_xml.find("object").to_list()
    w, h = my_xml.find("width", 0).to_int(), my_xml.find("height", 0).to_int()
    bboxes = []
    for this_obj in objects:
        bbox = this_obj.find("bndbox", 0)
        msg = {
            "class": this_obj.find("name", 0).to_string(),
            "bbox": bbox.find(["xmin", "ymin", "xmax", "ymax"], 0).to_float(),
            "difficult": this_obj.find("difficult", 0).to_bool()
        }
        bboxes.append(msg)
    data = {
        "width": w,
        "height": h,
        "bboxes": bboxes
    }
    return data


class VOCDataset(Dataset):
    """
    VOC dataset class.
    """
    coco = None
    num_annos = 0
    idx = 0
    max_num_labels = 0
    min_ratio = 0.8
    coco_data = None

    def __init__(
        self,
        cfg: dict = None,
        img_size=(640, 640),
        preproc: BaseTransform = None,
        is_train=True,
        test=False,
        **kwargs
    ):
        super().__init__(img_size)

        self.test = test
        self.unused_image = 0

        self.data_dir = cfg.get("dataset_path")
        sets: dict = cfg.get("train" if is_train else "val" if not test else "test")
        # print(self.data_dir, sets.get("image_dir"), sets)
        self.train_dir = os.path.join(self.data_dir, sets.get("image_dir"))
        if not test:
            self.anno_dir = os.path.join(self.data_dir, sets.get("anno_dir"))
            self.anno_file = os.path.join(self.data_dir, sets.get("label"))

        self.names = cfg.get("names")

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

        print("loading VOC dataset...")
        t = time()
        self.annotation_list = self._load_voc_annotations()

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

    def get_anno_by_name(self, name: str):
        msg = {
            "image": os.path.join(self.train_dir, f"{os.path.basename(name)[:-4]}.{self.suffix}"),
            "annotations": []
        }
        num_labels = 0
        with open(name, 'r') as xmlf:
            img_data = decode_VOC(xmlf)
        img_w, img_h = img_data.get("width"), img_data.get("height")
        has_anno = False

        for box in img_data.get("bboxes"):

            x1, y1, x2, y2 = box.get("bbox")
            class_name = box.get("class")
            class_id = self.names.index(class_name)

            num_labels += 1
            msg["annotations"].append([x1, y1, x2, y2, class_id])   # xywh
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
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    iscrowd=0
                )
            self.num_annos += 1
        if not has_anno:
            self.unused_image += 1
        self.max_num_labels = max(self.max_num_labels, num_labels)
        msg["annotations"] = np.array(msg["annotations"])
        return msg

    def _load_voc_annotations(self):

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
            with open(self.anno_file, "r") as f:
                lines = f.read().split("\n")
            for self.idx, line in enumerate(lines):
                if not len(line):
                    continue
                if ' ' in line:
                    file, _ = line.split()
                else:
                    file = line
                anno_file = os.path.join(self.data_dir, self.anno_dir, file + ".xml")
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
