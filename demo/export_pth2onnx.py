import argparse
import os
import sys
import time
import yaml

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

from edgeyolo import EdgeYOLO
from edgeyolo.utils2.activations import SiLU
from loguru import logger
# from edgeyolo.utils2.general import set_logging


def get_args():
    parser = argparse.ArgumentParser("EdgeYOLO onnx-export parser")
    parser.add_argument('--weights', type=str, default='./edgeyolo.pth', help='weights path')
    parser.add_argument('--input-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--opset', type=int, default=11, help='opset version')

    return parser.parse_args()


def replace_module(module, replaced_module_type, new_module_type, replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model


if __name__ == '__main__':
    opt = get_args()

    save_dir = "output/export/onnx/"
    os.makedirs(save_dir, exist_ok=True)
    print(opt)
    # set_logging()
    t = time.time()

    opt.grid = True
    opt.input_size *= 2 if len(opt.input_size) == 1 else 1

    device = torch.device('cpu')
    exp = EdgeYOLO(None, opt.weights)
    model = exp.model
    model.eval()
    with torch.no_grad():
        model.fuse()
    labels = exp.class_names

    replace_module(model, nn.SiLU, SiLU)

    img = torch.zeros(opt.batch, 3, *opt.input_size).to(device)
    model.model[-1].export = not opt.grid
    model(img)  # dry run

    # ONNX export
    try:
        import onnx

        logger.info('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        file_name = save_dir + os.path.basename(opt.weights[:-len(opt.weights.split(".")[-1])-1])



        f = file_name + f'_{opt.input_size[0]}x{opt.input_size[1]}_batch{opt.batch}.onnx'  # filename
        model.eval()
        model.model[-1].concat = True
        input_names = ["input_0"]
        output_names = ["output_0"]   # , "output_1", "output_2"]
        torch.onnx.export(model,
                          img,
                          f,
                          verbose=False,
                          opset_version=opt.opset,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=None)
        logger.info("export end")
        logger.info("check")
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        if opt.simplify:
            try:
                import onnxsim
                logger.info('\nstart to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                logger.error(f'Simplifier failure: {e}')
        else:
            logger.warning("simplify disabled. It is recommend to use "
                           "command\"--simplify\" to simplify your export model.")

        onnx.save(onnx_model, f)
        logger.info('ONNX export success, saved as %s' % f)

        with open(file_name + f"_{opt.input_size[0]}x{opt.input_size[1]}_batch{opt.batch}.yaml", "w") as fp:
            yaml.dump({
                "input_name": input_names,
                "output_name": output_names,
                "names": labels,
                "img_size": opt.input_size,
                "batch_size": opt.batch,
                "pixel_range": 255,         # input image pixel value range: 0-1 or 0-255
                "obj_conf_enabled": True,   # Edge-YOLO use cls conf and obj conf
            }, fp, yaml.Dumper)
            logger.info(f"params saved to {file_name}_{opt.input_size[0]}x{opt.input_size[1]}_batch{opt.batch}.yaml")

        print("")
        logger.info("############# - msg - ##############")
        logger.info(f"input names   : {input_names}")
        logger.info(f"output names  : {output_names}")
        try:
            logger.info(f"output shape  : {model(img).shape}")
        except AttributeError:
            logger.info(f"output shape  : {[m.shape for m in model(img)]}")
        except Exception as e:
            logger.error(e)

        logger.info(f"img size      : {opt.input_size}")
        logger.info(f"batch size    : {opt.batch}")
        logger.info(f"names         : {labels}")

    except Exception as e:
        logger.error('ONNX export failure: %s' % e)

    # Finish
    logger.info('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
