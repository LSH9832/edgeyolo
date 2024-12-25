from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy as np
import os.path as osp
from glob import glob
import sys
import os

suffix_map = {
    "engine": "TensorRT",
    "trt": "TensorRT",
    "rknn": "RKNN",
    "mnn": "MNN",
    "om": "Ascend",
    "bin": "Horizon"
}

swap_map = {
    "tensorrt": (2, 0, 1),
    "mnn": (2, 0, 1),
    "rknn": None,
    "ascend": (2, 0, 1),
    "horizon": None
}

dtype_map = {
    "tensorrt": np.float32,
    "mnn": np.float32,
    "rknn": np.uint8,
    "ascend": np.float32,
    "horizon": np.uint8,
}

def list_strings(strings):
    # 将Python字符串列表转换为C的char**类型
    strings_array = (c_char_p * len(strings))()
    for i, string in enumerate(strings):
        strings_array[i] = c_char_p(string.encode('utf-8'))
    
    return cast(strings_array, POINTER(c_char_p))

class NoPrint:

    def __init__(self, flag=True):
        self.flag = flag

    def __enter__(self):
        if self.flag:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.flag:
            sys.stdout.close()
            sys.stdout = self._original_stdout


class LibBase:

    def __init__(self, model_path: str):
        self.suffix = None
        self._set(model_path)

        # print(self.dtype)

    def platform(self) -> str:
        plat_str = b"                  "
        self.yoloPlatform(plat_str)
        return plat_str.decode().split()[0]

    def _set(self, model_path: str):
        self.suffix = model_path.split(".")[-1].lower()
        if self.suffix not in suffix_map:
            print(f"[E] do not support model type: {self.suffix}")
            exit(-1)
        
        libPath = osp.join(
            osp.dirname(__file__), "lib", 
            f"libyoloInfer{suffix_map[self.suffix]}.so"
        )
        if osp.isfile(libPath):
            lib = CDLL(libPath)
        else:
            raise IOError(f"can not open {libPath}")

        self.setupYOLO = lib.setupYOLO
        self.setupYOLO.argtypes = [
            c_char_p,
            c_char_p,
            POINTER(c_char_p),     # char**
            c_int,
            c_int,
            c_int,
            ndpointer(dtype=np.int32),
            c_int,
            c_int
        ]
        self.setupYOLO.restype = c_void_p


        self.initYOLO = lib.initYOLO
        self.initYOLO.argtypes = [c_void_p]
        self.initYOLO.restype = c_bool

        self.isInit = lib.isInit
        self.isInit.argtypes = [c_void_p]
        self.isInit.restype = c_bool

        self.getNumClasses = lib.getNumClasses
        self.getNumClasses.argtypes = [c_void_p]
        self.getNumClasses.restype = c_int

        self.getNumArrays = lib.getNumArrays
        self.getNumArrays.argtypes = [c_void_p]
        self.getNumArrays.restype = c_int

        self.yoloSet = lib.set
        self.yoloSet.argtypes = [
            c_void_p,
            c_char_p,
            c_char_p
        ]

        self.yoloPlatform = lib.platform
        self.yoloPlatform.argtypes = [c_char_p]

        self.swap = swap_map[self.platform().lower()]
        self.dtype = dtype_map[self.platform().lower()]

        self.inference = lib.inference
        self.inference.argtypes = [
            c_void_p, 
            ndpointer(dtype=self.dtype),
            ndpointer(dtype=np.float32),
            c_float
        ]

        self.releaseYOLO = lib.releaseYOLO
        self.releaseYOLO.argtypes = [c_void_p]
