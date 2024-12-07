# from ctypes import cast, cdll, c_int, POINTER, c_float, CDLL, addressof, c_void_p, c_char_p, c_bool
import numpy as np
# from numpy.ctypeslib import ndpointer
# import os.path as osp
import cv2
# from .postprocess import postprocess
from .libbase import LibBase, list_strings, NoPrint

from time import time


def preproc(img, input_size, normalize=False, swap=(2, 0, 1), dtype=np.float32):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    if swap is not None:
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=dtype)
    elif padded_img.dtype != dtype:
        padded_img = padded_img.astype(dtype)
    
    if normalize:
        padded_img /= 255.
    return padded_img, r


class YOLO(LibBase):

    def __init__(
        self,
        modelFile,
        names,
        inputName, 
        outputNames, 
        imgW, 
        imgH,
        normalize=False,
        strides=None, 
        device=0,
        **kwargs
    ):
        super().__init__(modelFile)
        self.names = names
        self.size = [imgH, imgW]
        self.normalize = normalize
        if strides is None:
            strides = [8, 16, 32]
        strides = np.array(strides, dtype=np.int32)

        with NoPrint():
            self.yolo = self.setupYOLO(
                modelFile.encode("utf8"), 
                inputName.encode("utf8"), 
                list_strings(outputNames), 
                len(outputNames),
                imgW, imgH, 
                strides, len(strides), 
                device
            )
        for k, v in kwargs.items():
            # print(k, v)
            if k == "anchors":
                anchor_str = ""
                for i, stride_anchors in enumerate(v):
                    if i:
                        anchor_str += ";"
                    for j, anchor in enumerate(stride_anchors):
                        if j:
                            anchor_str += ","
                        anchor_str += f"{anchor[0]} {anchor[1]}"
                
                self.yoloSet(self.yolo, k.encode('utf8'), anchor_str.encode('utf8'))
            else:
                self.yoloSet(self.yolo, k.encode('utf8'), str(v).encode('utf8'))
        
        if not self.initYOLO(self.yolo):
            print("failed to init yolo, exit")
            exit(-1)

        # self.conf_thres = confThreshold
        # self.nms_thres = nmsThreshold

        self.num_arrays = self.getNumArrays(self.yolo)
        self.num_classes = self.getNumClasses(self.yolo)
        for i in range(len(names), self.num_classes):
            self.names.append(f"unknown_{i+1}")
        self.__preds = np.zeros([self.num_arrays * (self.num_classes+5)], dtype=np.float32)
        
    def infer(self, img: np.ndarray):

        t0 = time()
        pad_img, ratio = preproc(img, self.size, self.normalize, self.swap, self.dtype)
        t1 = time()

        self.inference(self.yolo, pad_img, self.__preds, ratio)
        t2 = time()

        # print(f" preprocess:{1000 * (t1 - t0):.3f}ms, infer:{1000 * (t2 - t1):.3f}ms", end="     ")
        result = self.__preds[1:1+int(self.__preds[0]) * 6].reshape([-1, 6])
        # result = postprocess(self.__preds, ratio, img.shape[:2], self.conf_thres, self.nms_thres)
        return result

    def release(self):
        self.releaseYOLO(self.yolo)