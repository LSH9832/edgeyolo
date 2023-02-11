from ..models import EdgeYOLO
from ..data.data_augment import preproc
from ..utils import postprocess, get_model_info
import torch
import numpy as np
from time import time


class Detector(EdgeYOLO):

    conf_thres = 0.25
    nms_thres = 0.5
    fuse = True
    cpu = False
    fp16 = False
    use_decoder = False

    def __init__(self, weight_file, **kwargs):
        super(Detector, self).__init__(None, weight_file)
        
        for k, v in kwargs.items():
            if hasattr(self, k):
                self.__setattr__(k, v)
            else:
                print(f"no keyword named {k}")

        print(get_model_info(self.model, self.input_size))

        if self.fuse:
            with torch.no_grad():
                self.model.fuse()
                print("After re-parameterization:", get_model_info(self.model, self.input_size))

        if not self.cpu:
            self.model.cuda(0)
            if self.fp16:
                self.model.half()
        self.model.eval()

    def __preprocess(self, imgs):
        pad_ims = []
        rs = []
        for img in imgs:
            pad_im, r = preproc(img, self.input_size)
            pad_ims.append(torch.from_numpy(pad_im).unsqueeze(0))
            rs.append(r)
        self.t0 = time()

        ret_ims = pad_ims[0] if len(pad_ims) == 1 else torch.cat(pad_ims)

        return ret_ims, rs

    def __postprocess(self, results, rs):
        # print(results, results.shape)

        outs = postprocess(results, len(self.class_names), self.conf_thres, self.nms_thres, True)
        for i, r in enumerate(rs):
            if outs[i] is not None:
                outs[i] = outs[i].cpu()
                outs[i][..., :4] /= r
        return outs
    
    def decode_outputs(self, outputs):
        dtype = outputs.type()
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.input_size, self.strides):
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
            imgs = [imgs]

        with torch.no_grad():
            inputs, ratios = self.__preprocess(imgs)
            if not self.cpu:
                inputs = inputs.cuda()
                if legacy:
                    inputs /= 255
                if self.fp16:
                    inputs = inputs.half()
            try:
                net_outputs = self.model(inputs)
            except:
                print(inputs.shape)
                raise
            if self.use_decoder:
                net_outputs = self.decode_outputs(net_outputs)
            outputs = self.__postprocess(net_outputs, ratios)
            self.dt = time() - self.t0

        return outputs

