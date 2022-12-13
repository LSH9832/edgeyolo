import yaml
import argparse
import os.path as osp
import os
from loguru import logger
import torch


def get_args():
    parser = argparse.ArgumentParser("EdgeYOLO onnx2tensorrt parser")
    parser.add_argument("-o", "--onnx", type=str, default="yolov7.onnx", help="ONNX file")
    parser.add_argument("-y", "--yaml", type=str, default="yolov7.yaml", help="export params file")
    parser.add_argument("-w", "--workspace", type=int, default=8, help="export memory workspace(GB)")
    parser.add_argument("--fp16", action="store_true", help="fp16")
    parser.add_argument("--int8", action="store_true", help="int8")
    parser.add_argument("--best", action="store_true", help="best")
    parser.add_argument("-d", "--dist-path", type=str, default="export_output/tensorrt")
    parser.add_argument("--batch-size", type=int, default=0, help="batch-size")
    return parser.parse_args()


def main():
    args = get_args()
    
    assert osp.isfile(args.onnx), f"No such file named {args.onnx}."
    assert osp.isfile(args.yaml), f"No such file named {args.yaml}."

    os.makedirs(args.dist_path, exist_ok=True)

    name = args.onnx.replace("\\", "/").split("/")[-1][:-len(args.onnx.split(".")[-1])]

    engine_file = osp.join(args.dist_path, name + "engine").replace("\\", "/")
    pt_file = osp.join(args.dist_path, name + "pt").replace("\\", "/")
    cls_file = osp.join(args.dist_path, name + "txt").replace("\\", "/")
    params = yaml.load(open(args.yaml).read(), yaml.Loader)
    command = f"trtexec --onnx={args.onnx}" \
              f"{' --fp16' if args.fp16 else ' --int8' if args.int8 else ' --best' if args.best else ''} " \
              f"--saveEngine={engine_file} --workspace={args.workspace*1024} " \
              f"--batch={args.batch_size if not args.batch_size > 0 else params['batch_size'] if 'batch_size' in params else 1}"

    logger.info("start converting onnx to tensorRT engine file.")
    os.system(command)

    if not osp.isfile(engine_file):
        logger.error("tensorRT engine file convertion failed.")
        return

    logger.info(f"tensorRT engine saved to {engine_file}")

    try:
        data = {
            "model": {
                "engine": bytearray(open(engine_file, "rb").read()),
                "input_names": params["input_name"],
                "output_names": params["output_name"]
            },
            "names": params["names"],
            "img_size": params["img_size"],
            "batch_size": params["batch_size"]
        }
        class_str = ""
        for name in params["names"]:
            class_str += name + "\n"
        with open(cls_file, "w") as cls_f:
            cls_f.write(class_str[:-1])
            logger.info(f"class names txt pt saved to {cls_file}")
        torch.save(data, pt_file)
        logger.info(f"tensorRT pt saved to {pt_file}")
    except Exception as e:
        logger.error(f"convert2pt error: {e}")


if __name__ == '__main__':
    main()
