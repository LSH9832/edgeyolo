import os
import os.path as osp
from glob import glob
import yaml
import random
import numpy as np
import torch
from time import time
import cv2

from ..data.data_augment import preproc
from ..utils import postprocess
# from ..detect import draw


class RKNNExporter:

    def __init__(self, onnx_file, rknn_file, platform, dataset=None, num=512,
                 train=False, use_all=False, input_size=(640, 640), temp_file="calib_temp.txt"):
        self.onnx_file = onnx_file
        self.rknn_file = rknn_file
        self.rand_img = None
        self.temp_file = temp_file

        self.conf_thres = 0.25
        self.nms_thres = 0.45
        self.class_names = []

        self.input_size = input_size
        self.batch_size = 1

        from rknn.api import RKNN
        self.rknn = RKNN(verbose=True)
        self.rknn.config(target_platform=platform)
        ret = self.rknn.load_onnx(model=onnx_file)
        if ret != 0:
            print('Load model failed!')
            exit(ret)

        self.create_dataset(dataset, num, train, use_all, temp_file)
        ret = self.rknn.build(do_quantization=dataset is not None, dataset=temp_file if dataset is not None else None)
        if ret != 0:
            print('Build model failed!')
            exit(ret)

    def create_dataset(self, dataset, num, train, use_all, temp_file):
        if dataset is None:
            return

        with open(dataset) as yamlf:
            dataset_cfg: dict = yaml.load(yamlf, yaml.Loader)

        if use_all:
            imgs_path = [os.path.join(dataset_cfg.get("dataset_path"), dataset_cfg.get("train").get("image_dir")),
                         os.path.join(dataset_cfg.get("dataset_path"), dataset_cfg.get("val").get("image_dir"))]
        else:
            sub_dataset = "train" if train else "val"
            imgs_path = [os.path.join(dataset_cfg.get("dataset_path"), dataset_cfg.get(sub_dataset).get("image_dir"))]

        img_files = []
        suffix = dataset_cfg.get("kwargs").get("suffix")

        # from loguru import logger

        for path in imgs_path:
            # logger.info(osp.join(path, f"*.{suffix}"))
            img_files.extend(glob(osp.join(path, f"*.{suffix}")))

        random.shuffle(img_files)

        img_files = img_files[:num]
        # import cv2
        self.rand_img = cv2.imread(img_files[0])

        # cv2.imshow("test", self.rand_img)
        # cv2.waitKey(0)

        temp_file_str = ""
        for img_file in img_files:
            temp_file_str += f"{img_file}\n"

        temp_file_str = temp_file_str[:-1]

        with open(temp_file, "w") as f:
            f.write(temp_file_str)

    def convert(self, input_img, batch_size):
        self.input_size = input_img.shape[:2]
        self.batch_size = batch_size
        ret = self.rknn.export_rknn(self.rknn_file)
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        print('done')

        # get output size
        ret = self.rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        # print('done')

        # print('--> Running model')
        outputs = self.rknn.inference(inputs=[np.array([input_img] * batch_size)])

        # print(batch_size)

        # print(input_img.shape)
        # print(type(outputs))
        if outputs is not None:
            # print(len(outputs))
            for i in range(len(outputs)):
                output = outputs[i]
                if output is not None:
                    print("output", i, output.shape)
        
        if osp.isfile(self.temp_file):
            os.remove(self.temp_file)
        # return
        # output = outputs[0]
        # print(f"output shape: {output.shape}")
        # self.class_names = [str(i) for i in range(output.shape[-1] - 5)]
        #
        # result = self(self.rand_img)
        # img = draw(self.rand_img, result, self.class_names, 2)
        #
        # cv2.imshow("rknn test", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite("rknn_result.jpg", img)


    def __preprocess(self, imgs):
        pad_ims = []
        rs = []
        for img in imgs:
            pad_im, r = preproc(img, self.input_size, (0, 1, 2))
            pad_ims.append(pad_im)
            rs.append(r)
        assert len(pad_ims) == self.batch_size, "batch size not match!"
        self.t0 = time()

        return np.array(pad_ims), rs   # .float()

    def __postprocess(self, results, rs):
        # print(results, results.shape)
        # import torch
        print("shape: ", results[0].shape)
        max_num = np.max(results[0][..., 5:])
        print(max_num, 1. / (1 + np.exp(-max_num)))
        result = torch.from_numpy(results[0])
        result[..., 4:] = (result[..., 4:]).sigmoid()
        outs = postprocess(torch.from_numpy(results[0]), len(self.class_names), self.conf_thres, self.nms_thres, True)

        for i, r in enumerate(rs):
            if outs[i] is not None:
                outs[i][..., :4] /= r
                outs[i] = outs[i].cpu()
        return outs

    def __call__(self, imgs, legacy=False):
        if isinstance(imgs, np.ndarray):
            # print(imgs)
            imgs = [imgs]

        input_imgs, rs = self.__preprocess(imgs)

        net_outputs = self.rknn.inference(inputs=[input_imgs])

        outputs = self.__postprocess(net_outputs, rs)
        self.dt = time() - self.t0

        return outputs


