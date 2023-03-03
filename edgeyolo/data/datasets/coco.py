import os
import time

from loguru import logger

import cv2
import numpy as np
import datetime
import pickle
import json
from pycocotools.coco import COCO

from .datasets_wrapper import Dataset


def remove_useless_info(coco, use_segments=True):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if not use_segments:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    coco = None
    annotations = None
    imgs = None
    segm_len = 0
    ids = []
    class_ids = [i for i in range(80)]
    _classes = [str(i) for i in range(80)]
    max_num_labels = 0

    def __init__(
        self,
        data_dir=None,
        json_file="",
        train_dir="",
        img_size=(640, 640),
        preproc=None,
        cache=False,
        load_segment=True,
        is_train=True,
        cfg: dict = None,
        test=False,
        **kwargs
    ):

        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int or tuple): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if cfg is None:
            self.data_dir = data_dir
            self.json_file = json_file
            self.train_dir = train_dir
            self.load_segm = load_segment
        else:
            self.data_dir = cfg.get("dataset_path")
            sets: dict = cfg.get("train" if is_train else "val" if not test else "test")
            self.json_file = os.path.join(self.data_dir, sets.get("label"))
            self.train_dir = os.path.join(self.data_dir, sets.get("image_dir"))
            self.load_segm = cfg.get("segmentaion_enabled")

        self.use_cache = kwargs.get("use_cache") or False
        self.img_size = img_size
        self.preproc = preproc
        self.is_train = is_train
        self.test = test
        self.names = cfg.get("names")

        self.load()

        if cache:
            self._cache_images()

    def load(self):

        self.cache_file = \
            cache_file = os.path.join(self.data_dir,
                                      f"{'train' if self.is_train else 'val' if not self.test else 'test'}_"
                                      f"cache{'_with_seg' if self.load_segm else ''}.edgeyolo").replace("\\", "/")
        self.cached = True
        if os.path.isfile(cache_file) and self.use_cache:
            print("loading COCO dataset...")
            print(f"loading cache from {cache_file}.")
            t0 = time.time()

            with open(cache_file, "rb") as f:
                self.coco, self.annotations, self.segm_len, self.max_num_labels = pickle.load(f)

            self.ids = self.coco.getImgIds()
            self.class_ids = sorted(self.coco.getCatIds())
            cats = self.coco.loadCats(self.coco.getCatIds())
            self._classes = tuple([c["name"] for c in cats])

            print(f"DONE(t={time.time() - t0:.2f}s)")

        else:
            print("loading COCO dataset...")
            self.coco = COCO(self.json_file)
            self.cached = False
            try:
                remove_useless_info(self.coco, self.load_segm)
            except:
                pass

            self.ids = self.coco.getImgIds()
            self.class_ids = sorted(self.coco.getCatIds())
            cats = self.coco.loadCats(self.coco.getCatIds())
            self._classes = tuple([c["name"] for c in cats])

            self.annotations = self._load_coco_annotations()

            # try:
            #     with open(cache_file, "wb") as cachef:
            #         pickle.dump((self.coco, self.annotations, self.segm_len, self.max_num_labels), cachef)
            # except:
            #     pass

        if self.load_segm:
            print("max len segmentation:", self.segm_len)

        if self.preproc is not None:
            print(f"max label number in one image: {self.max_num_labels}")
            self.preproc.set_max_labels(max(self.max_num_labels * 5, 50))

    def save_cache(self):
        if self.cached:
            return
        with open(self.cache_file, "wb") as cachef:
            pickle.dump((self.coco, self.annotations, self.segm_len, self.max_num_labels), cachef)

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        try:
            del self.imgs
        except:
            pass

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.train_dir.split('/')[-1] + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []

        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]

                objs.append(obj)

        num_objs = len(objs)
        self.max_num_labels = max(self.max_num_labels, num_objs)

        res = np.zeros((num_objs, 5))
        segms = [] if self.load_segm else None

        max_wh = max(height, width)
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            if self.load_segm:
                segm = [np.array([[edge[i], edge[i+1]]
                                  for i in range(0, len(edge), 2)]) / max_wh
                        for edge in obj["segmentation"]]
                for edge in obj["segmentation"]:
                    self.segm_len = max(self.segm_len, len(edge))
                segms.append(segm)

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name, segms)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)

        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        # print(file_name)

        img_file = os.path.join(self.train_dir, file_name).replace('\\', '/')

        img = cv2.imread(img_file)
        assert img is not None, f"{img_file} is not exist or broken!"
        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        # rectangles, img_info, resized_info, filename, segments
        res, img_info, resized_info, file_name, segms = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, np.array([id_]), segms

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


class COCOCreator:
    """
    create a dataset with coco format
    """

    def __init__(self, fp=None):
        """
        :param fp: json file
        """
        self.info = {
            'description': "UnNamed",
            'url': "",
            'version': "1.0",
            'year': datetime.datetime.now().year,
            'contributor': "UnNamed",
            'date_created': '%s/%s/%s' % (str(datetime.datetime.now().year),
                                          str(datetime.datetime.now().month).zfill(2),
                                          str(datetime.datetime.now().day).zfill(2))
        }
        self.lic = []
        self.images = []
        self.annotations = []
        self.categories = []
        self.categorie_num = {}

        if isinstance(fp, str):
            assert os.path.exists(fp)
            fp = open(fp)
        self.load(fp)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lic
        del self.images
        del self.annotations
        del self.categories
        del self.categorie_num
        return False

    def son_of(self, coco_parent, name="son"):
        """
        Copy messages from parent coco dataset without images and annotations
        :param coco_parent: parent coco dataset
        :param name: name of this dataset, /train/val/test
        """
        self.info = coco_parent.get_info()
        # print(parent.get_info())

        self.info["description"] += "_%s" % name
        self.lic = coco_parent.get_license()
        self.categories = coco_parent.get_categories()
        self.categorie_num = {idx: 0 for idx in coco_parent.get_category_num()}
        # print(coco_parent.get_category_num())

    def get_info(self):
        return self.info.copy()

    def get_license(self):
        return self.lic.copy()

    def get_images(self):
        return self.images

    def get_categories(self):
        return self.categories.copy()

    def get_category_num(self):
        return self.categorie_num.copy()

    def _count_category_num(self):
        """
        count and update number of each category
        """
        # print("counting categories")
        self.categorie_num = {}
        for anno in self.annotations:
            if anno['category_id'] in self.categorie_num:
                self.categorie_num[anno['category_id']] += 1
            else:
                self.categorie_num[anno['category_id']] = 1

    def _image_id_exists(self, image_id):
        """
        whether this image id exists
        :param image_id: image id
        """
        flag = False
        for image in self.images:
            if image["id"] == image_id:
                flag = True
                break
        return flag

    def _get_image_data_by_id(self, image_id):
        """
        get image data by its id
        :param image_id:
        :return: image data(dict)
        """
        for image in self.images:
            if image["id"] == image_id:
                return image
        return None

    def _get_category_id_by_name(self, name, add_mode=True):
        """
        get category id by its name
        :param name: category name
        :return: category id(int)
        """
        for this_category in self.categories:
            if this_category['name'] == name:
                return this_category["id"]
        if add_mode:
            self.add_category(name)
            return self._get_category_id_by_name(name, False)
        else:
            assert add_mode, "category %s not Found" % name

    def _get_name_by_category_id(self, cid):
        for c in self.categories:
            if c["id"] == cid:
                return c["name"]
        return None

    def total_image_number(self):
        return len(self.images)

    def total_annotation_number(self):
        # TODO
        return len(self.annotations)

    def load(self, fp):
        """
        load data from json file
        :param fp: file
        """

        if fp is not None:
            from time import time
            print("loading json...")
            t0 = time()
            data = json.load(fp)
            print("loading time: %.2fs" % (time()-t0))
            self.info = data["info"]
            self.lic = data["licenses"]
            self.images = data["images"]
            self.annotations = data["annotations"]
            self.categories = data["categories"]
            self._count_category_num()

    def change_info(self, data_name, version="1.0", url="", author=""):
        """
        change information of this dataset
        :param data_name: name of this dataset
        :param version: version of this dataset
        :param url: url link of this dataset
        :param author: author of this dataset
        """
        self.info = {
            'description': data_name,
            'url': url,
            'version': version,
            'year': datetime.datetime.now().year,
            'contributor': author,
            'date_created': '%s/%s/%s' % (str(datetime.datetime.now().year),
                                          str(datetime.datetime.now().month).zfill(2),
                                          str(datetime.datetime.now().day).zfill(2))
        }

    def add_license(self, name, url=""):
        """
        add license of this dataset
        :param name: name of this license
        :param url: url link of this license
        """
        self.lic.append({
            'url': url,
            'id': len(self.lic) + 1,
            'name': name
        })

    def add_category(self, name, supercategory=None):
        """
        add category of this dataset
        :param name: category name
        :param supercategory: supercategory name
        """
        self.categories.append({
            'supercategory': supercategory,
            'id': len(self.categories),
            'name': name
        })
        self.categorie_num[len(self.categories) - 1] = 0

    def load_categories(self, names):
        """
        load category from txt file
        :param names: file name or dict
        """
        if names is not None:

            if isinstance(names, str):
                with open(names) as fp:
                    classes = fp.read().split("\n")
            else:
                classes = names
            assert isinstance(classes, list)
            for this_class in classes:
                while this_class.endswith(" "):
                    this_class = this_class[:-1]
                if len(this_class):
                    this_class = this_class.split(":")
                    self.add_category(name=this_class[0], supercategory=this_class[-1])
        self.categorie_num = {}

    def add_image(
            self,
            image_id,
            file_name,
            width,
            height,
            date_captured=None,
            license_id=1,
            url="",
            flickr_url=None
    ):
        """
        add image data to this dataset
        :param image_id: image id
        :param file_name: file name e.g: 00001.jpg
        :param width: image width
        :param height: image height
        :param date_captured: e.g 2022-02-22 22:22:22
        :param license_id: license id
        :param url: image url
        :param flickr_url: image flickr url
        """
        assert not self._image_id_exists(image_id), "Image ID %d already exists!" % image_id
        self.images.append({
            'license': license_id,
            'file_name': file_name,
            'coco_url': url,
            'height': height,
            'width': width,
            'date_captured': str(datetime.datetime.now()).split(".")[0] if date_captured is None else date_captured,
            'flickr_url': url if flickr_url is None else flickr_url,
            'id': image_id
        })

    def add_annotation(
            self,
            image_id,
            anno_id,
            category_id,
            bbox=None,
            segmentation=None,
            area=None,
            iscrowd=0
    ):
        """
        add annotation of any image exists in this dataset
        :param image_id: image id
        :param anno_id: annotation id
        :param category_id: category id
        :param bbox: bounding box [xmin, ymin, w, h]
        :param segmentation: segmentation [[x00, y00, x01, y01, ....], [x10, y10, x11, y11, ....], ....]
        :param area: area of segmentation if segmentation is not empty else area of bounding box
        :param iscrowd: is crowd
        """
        assert bbox or segmentation, "bbox or segmentation is required"
        assert self._image_id_exists(image_id), "Image ID %d does not exist!" % image_id

        if bbox is None:
            bbox = []
        if segmentation is None:
            segmentation = []
        if area is None and len(bbox):
            area = bbox[2] * bbox[3]

        self.annotations.append({
            'segmentation': segmentation,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': image_id,
            'bbox': bbox,
            'category_id': category_id,
            'id': anno_id
        })
        if category_id in self.categorie_num:
            self.categorie_num[category_id] += 1
        else:
            self.categorie_num[category_id] = 1

    def to_json(self):
        return {
            'info': self.info,
            'licenses': self.lic,
            'images': self.images,
            'annotations': self.annotations,
            'categories': self.categories
        }

    def save(self, file_name: str = None):
        """
        save data to a json file
        :param file_name: file name with path
        """
        file_name = self.info["description"] if file_name is None else file_name
        file_name = file_name if file_name.endswith(".json") else "%s.json" % file_name
        json.dump(self.to_json(), open(file_name, "w"))
        print("coco annotation saved to %s." % os.path.abspath(file_name).replace("\\", "/"))

    def show_each_category_num(self, width=50, simple=False):
        """
        show number of each category in a table
        :param width: tabel width
        :param simple: show simple table
        """
        if simple:
            categories_str = ""
            for cate in self.categories:
                categories_str += cate["name"].ljust(width//3 * 2) + \
                                  str(self.categorie_num[cate["id"] - 1]
                                      if (cate["id"] - 1) in self.categorie_num else 0).rjust(width - width//3*2) + "\n"
            return f"""
{"=" * width}
Category Count
{"%s images" % str(len(self.images)).rjust(10)}
{"%s annotations" % str(len(self.annotations)).rjust(10)}
{"=" * width}
{categories_str}{"=" * width}"""
        width = max(28, int(width))
        head = "╒%s╕\n" \
               "│%sCategory Count%s│\n" \
               "╞%s╡\n" % ("═" * width, " " * int((width - 14) / 2), " " * int((width - 14) / 2), "═" * width)
        msg = ""
        msgs = ["%s images" % str(len(self.images)).rjust(10),
                "%s annotations" % str(len(self.annotations)).rjust(10)]
        for this_msg in msgs:
            msg += "│%s│\n" % this_msg.ljust(width)

        neck = "╞%s╤%s╡\n" % ("═" * (width - 1 - int(width / 3)), "═" * int(width / 3))



        body = ""
        for cate in self.categories:
            body += "│%s│%s│\n" % (
                cate["name"].ljust(width - 1 - int(width / 3)),
                str(self.categorie_num[cate["id"] - 1] if (cate["id"] - 1) in self.categorie_num else 0).rjust(int(width / 3))
            )
            # print("│%s│%s│" % (
            #     cate["name"].ljust(width - 1 - int(width / 3)),
            #     str(self.categorie_num[cate["id"] - 1] if (cate["id"] - 1) in self.categorie_num else 0).rjust(int(width / 3))
            # ))

            if self.categories.index(cate) < len(self.categories) - 1:
                body += "├%s┼%s┤\n" % ("─" * (width - 1 - int(width / 3)), "─" * int(width / 3))
                # print("├%s┼%s┤" % ("─" * (width - 1 - int(width / 3)), "─" * int(width / 3)))
            else:
                body += "╘%s╧%s╛" % ("═" * (width - 1 - int(width / 3)), "═" * int(width / 3))
                # print("╘%s╧%s╛" % ("═" * (width - 1 - int(width / 3)), "═" * int(width / 3)))

        show_str = head + msg + neck + body
        print()
        print(show_str)
        print()

        return show_str
