from ..data.data_augment import preproc
from ..utils import postprocess
import torch
import numpy as np
from loguru import logger
import os
from time import time


class OnnxDetector:
    pass


class TRTDetector:

    strides = [8, 16, 32]

    def __init__(self, weight_file, conf_thres, nms_thres, *args, **kwargs):
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        import torch2trt
        logger.info(f"loading weights from {weight_file}")
        self.model = torch2trt.TRTModule()
        self.model.eval()
        self.model.cuda()
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        ckpt = torch.load(weight_file, map_location="cpu")
        
        self.use_decoder = kwargs.get("use_decoder") or False
        
        # for k in ckpt:
        #     print(k)
        self.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        self.class_names = ckpt["names"] if "names" in ckpt else [str(i) for i in range(80)]
        self.input_size = ckpt["img_size"] if "img_size" in ckpt else kwargs["input_size"] if "input_size" in kwargs else [640, 640]
        if isinstance(self.input_size, int):
            self.input_size = [self.input_size] * 2
        # print(self.input_size)
        self.batch_size = ckpt["batch_size"] if "batch_size" in ckpt else 1

        x = torch.ones([self.batch_size, 3, *self.input_size]).cuda()
        logger.info(f"tensorRT input shape: {x.shape}")
        logger.info(f"tensorRT output shape: {self.model(x).shape[-3:]}")
        logger.info("tensorRT model loaded")

    def __preprocess(self, imgs):
        pad_ims = []
        rs = []
        for img in imgs:
            pad_im, r = preproc(img, self.input_size)
            pad_ims.append(torch.from_numpy(pad_im).unsqueeze(0))
            rs.append(r)
        assert len(pad_ims) == self.batch_size, "batch size not match!"
        self.t0 = time()
        ret_ims = pad_ims[0] if len(pad_ims) == 1 else torch.cat(pad_ims)
        return ret_ims.float(), rs

    def __postprocess(self, results, rs):
        # print(results, results.shape)

        outs = postprocess(results, len(self.class_names), self.conf_thres, self.nms_thres, True)

        for i, r in enumerate(rs):
            if outs[i] is not None:
                outs[i][..., :4] /= r
                outs[i] = outs[i].cpu()
        return outs
    
    def decode_outputs(self, outputs):
        dtype = outputs.type()
        grids = []
        strides = []
        
        for stride in self.strides:
            hsize, wsize = self.input_size[0] / stride, self.input_size[1] / stride
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride, dtype=torch.long))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def __call__(self, imgs, legacy=False):
        if isinstance(imgs, np.ndarray):
            # print(imgs)
            imgs = [imgs]

        with torch.no_grad():

            inputs, ratios = self.__preprocess(imgs)

            inputs = inputs.cuda()
            if legacy:
                inputs /= 255
            # if self.fp16:
            #     inputs = inputs.half()

            net_outputs = self.model(inputs)
            # print(net_outputs.shape)
            if len(net_outputs.shape) == 4:
                net_outputs = net_outputs[0]
            if self.use_decoder:
                net_outputs = self.decode_outputs(net_outputs)
            outputs = self.__postprocess(net_outputs, ratios)
            self.dt = time() - self.t0

        return outputs
