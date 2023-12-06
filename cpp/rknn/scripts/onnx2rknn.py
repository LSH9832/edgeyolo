from rknn.api import RKNN
import argparse
import os.path as osp
from glob import glob
import yaml
import random


PLATFORM = "rk3588"


def get_args():
    parser = argparse.ArgumentParser("EdgeYOLO RKNN model converter parser")

    parser.add_argument("-o", "--onnx", type=str, default="edgeyolo_tiny_lrelu_coco.onnx", help="onnx model")
    parser.add_argument("-r", "--rknn", type=str, default="edgeyolo_tiny_lrelu_coco.rknn", help="rknn model")

    parser.add_argument("-p", "--platform", type=str, default=PLATFORM)

    # quantize
    parser.add_argument("--no-quantize", action="store_true", help="do not quantize")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="dataset img path")
    parser.add_argument("--suffix", type=str, default="jpg", help="dataset image suffix")
    parser.add_argument("--num", type=int, default=512, help="number of calib files")

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)
    # pre-process config
    print('--> Config model')
    rknn.config(target_platform=args.platform)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')

    if args.dataset is not None:
        temp_file = "temp_calib.txt"
        with open(args.dataset) as yamlf:
            dataset_cfg: dict = yaml.load(yamlf, yaml.Loader)

        img_files = glob(osp.join(args.dataset, f"*.{args.suffix}"))
        random.shuffle(img_files)

        img_files = img_files[:args.num]

        temp_file_str = ""
        for img_file in img_files:
            temp_file_str += f"{img_file}\n"

        temp_file_str = temp_file_str[:-1]

        with open(temp_file, "w") as f:
            f.write(temp_file_str)

    ret = rknn.build(do_quantization=not args.no_quantize, dataset=None if args.no_quantize else args.dataset)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(args.rknn)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')