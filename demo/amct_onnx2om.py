#! /usr/bin/python3.7.5
# -*- coding: UTF-8 -*-
import os
import cv2

import numpy as np
import os.path as osp
import onnxruntime as ort
import amct_onnx as amct

from glob import glob


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputModel", type=str, required=True, help="input exist onnx model")
    parser.add_argument("-o", "--outputDir", type=str, default=None, help="output dir name")
    parser.add_argument("-b", "--batch", type=int, default=1, help="number of calib batch")
    parser.add_argument("-d", "--dataset", type=str, default="", 
                        help="calib image dataset path")
    parser.add_argument("-s", "--suffix", type=str, default="jpg", nargs="+", help="suffix")
    parser.add_argument("-f", "--framework", type=int, default=5)
    parser.add_argument("--soc", type=str, default="Ascend310", help="ascend soc version")
    parser.add_argument("--shuffle", action="store_true", help="shuffle your calib images")
    parser.add_argument("--fusion", action="store_true", help="fusion layers")
    parser.add_argument("--auto-tune", action="store_true", help="auto tune")
    parser.add_argument("--no-offset", action="store_true")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--om", action="store_true", help="export om model, ascend-cann-toolkit required")

    return parser.parse_args()


def preproc(img, input_size, swap=(2, 0, 1)):
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

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def prepare_image_input(images, shape):
    batch = len(images)
    ret = np.zeros([batch, 3, *shape], dtype=np.float32)
    for i, image_path in enumerate(images):
        pad_im, _ = preproc(cv2.imread(image_path), shape)
        ret[i] = pad_im
    return ret


def calib_forward(onnx_model, dataset_path: str, 
                  batch_size=1, num_imgs=32, suffix="jpg", calib=True):

    if dataset_path.endswith(".yaml"):
        import yaml
        cfg = yaml.load(open(dataset_path), yaml.SafeLoader)
        suffix = cfg["kwargs"]["suffix"]
        dataset_path = osp.join(cfg["dataset_path"], cfg["val"]["image_dir"])

    
    if isinstance(suffix, str):
        suffix = [suffix]
    images = []
    [images.extend(glob(osp.join(dataset_path, f"*.{s}"))) for s in suffix]

    if args.shuffle:
        np.random.shuffle(images)
    assert len(images), "number of image should not be zero, check your image path!"

    if calib:
        # amct.AMCT_SO.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ort_session = ort.InferenceSession(onnx_model, amct.AMCT_SO)
    else:
        ort_session = ort.InferenceSession(onnx_model)
    input_name = ort_session.get_inputs()[0].name
    _, _, height, width = ort_session.get_inputs()[0].shape

    iterations = int(round(float(num_imgs) / batch_size))
    for i in range(iterations):
        ims = images[i * batch_size : (i + 1) * batch_size]
        while len(ims) < batch_size:
            ims.append(images[np.random.randint(0, len(images))])
        input_batch = prepare_image_input(
            ims,
            (height, width)
        )
        ort_session.run(None, {input_name: input_batch})
        if calib:
            print(f"calib: run iter: {i+1}/{iterations}")


def amct_onnx(args):
    
    model_file = args.inputModel
    assert model_file.endswith(".onnx")
    if args.outputDir is None:
        args.outputDir = osp.join(osp.dirname(osp.abspath(model_file)), "amct_output")

    PATH = args.outputDir
    os.makedirs(PATH, exist_ok=True)

    

    TMP = osp.join(PATH, 'tmp')
    config_json_file = osp.join(TMP, 'config.json')
    skip_layers = []

    session = ort.InferenceSession(model_file)
    input_name = session.get_inputs()[0].name
    batch_num, channel, height, width = session.get_inputs()[0].shape

    if args.om and args.no_quant:
        onnx_file = model_file
        f = osp.basename(model_file).split('.')[0]


        fusion_switch_file = osp.join(PATH, 'fusion_switch.cfg')
        if args.fusion:
            open(fusion_switch_file, "w", encoding="utf8").write("""RequantFusionPass:off
TbeConvDequantVaddReluQuantFusionPass:off
TbeConvDequantQuantFusionPass:off
TbePool2dQuantFusionPass:off
""")

        print("start convert to om file")
        # command = f"/usr/local/Ascend/ascend-toolkit/5.0.mdc300/atc/bin/atc " \
        auto_tune_str = "--auto_tune_mode='GA' "
        command = "atc " + \
                  f"--input_shape='{input_name}:{batch_num},{channel},{height},{width}' " + \
                  f"--check_report={osp.join(PATH, 'network_analysis.report')} " + \
                  f"--input_format=NCHW " + \
                  f"{auto_tune_str if args.auto_tune else ''}" + \
                  f"--output='{osp.join(PATH, f)}' " + \
                  f"--soc_version={args.soc} --framework={args.framework} " + \
                  f"--model='{onnx_file}' " + \
                  (f"--fusion_switch_file={fusion_switch_file}" if args.fusion else "")
        print(command)
        os.system(command)
        return
    
    # print(batch_num)
    calib_batch = args.batch
    if isinstance(batch_num, str) or batch_num < 1:
        batch_num = 1

    # calib_forward(model_file, args.dataset, batch_num, args.num_imgs, args.suffix, calib=False)

    amct.create_quant_config(config_file=config_json_file,
                             model_file=model_file,
                             skip_layers=skip_layers,
                             batch_num=calib_batch,
                             activation_offset=not args.no_offset,
                             config_defination=None)

    scale_offset_record_file = osp.join(TMP, 'scale_offset_record.txt')
    modified_model = osp.join(TMP, 'modified_model.onnx')
    amct.quantize_model(config_file=config_json_file,
                        model_file=model_file,
                        modified_onnx_file=modified_model,
                        record_file=scale_offset_record_file)
    
    print("calib forward")
    calib_forward(modified_model, args.dataset, batch_num, calib_batch * batch_num, args.suffix)

    print("save model")
    f = osp.basename(model_file).split('.')[0]
    amct.save_model(modified_model, scale_offset_record_file, osp.join(PATH, f))

    if args.om:
        onnx_file = glob(osp.join(PATH, "*deploy*onnx"))
        assert len(onnx_file) == 1
        onnx_file = onnx_file[0]

        fusion_switch_file = osp.join(PATH, 'fusion_switch.cfg')
        if args.fusion:
            open(fusion_switch_file, "w", encoding="utf8").write("""RequantFusionPass:off
TbeConvDequantVaddReluQuantFusionPass:off
TbeConvDequantQuantFusionPass:off
TbePool2dQuantFusionPass:off
""")

        print("start convert to om file")
        # command = f"/usr/local/Ascend/ascend-toolkit/5.0.mdc300/atc/bin/atc " \
        auto_tune_str = "--auto_tune_mode='GA' "
        command = "atc " + \
                  f"--input_shape='{input_name}:{batch_num},{channel},{height},{width}' " + \
                  f"--check_report={osp.join(PATH, 'network_analysis.report')} " + \
                  f"--input_format=NCHW " + \
                  f"{auto_tune_str if args.auto_tune else ''}" + \
                  f"--output='{osp.join(PATH, f)}' " + \
                  f"--soc_version={args.soc} --framework={args.framework} " + \
                  f"--model='{onnx_file}' " + \
                  (f"--fusion_switch_file={fusion_switch_file}" if args.fusion else "")
        print(command)
        os.system(command)


if __name__ == '__main__':
    args = get_args()
    amct_onnx(args)
