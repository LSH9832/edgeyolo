import os
import json
import yaml
import torch
import argparse
import numpy as np

from loguru import logger

from edgeyolo import EdgeYOLO


def get_args():
    parser = argparse.ArgumentParser("EdgeYOLO Export Parser")

    # basic
    parser.add_argument("--weights", type=str, default="./weights/edgeyolo_tiny_coco.pth")
    parser.add_argument("--input-size", type=int, nargs="+", default=[640, 640])
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')

    # onnx
    parser.add_argument("--onnx", action="store_true", help="save onnx model(if tensorrt and torch2trt are installed)")
    parser.add_argument("--onnx-only", action="store_true", help="(if tensorrt and torch2trt are not installed)")
    parser.add_argument("--no-simplify", action="store_true", help="do not simplify models(not recommend)")
    parser.add_argument("--opset", type=int, default=11, help="onnx opset")

    # parser.add_argument("--relu", action="store_true", help="replace silu with relu")

    # tensorrt
    parser.add_argument("--trt", action="store_true", help="save tensorrt models")
    parser.add_argument("-w", '--workspace', type=float, default=8, help='max workspace size(GB)')

    ## fp16 quantization
    parser.add_argument("--no-fp16", action="store_true", help="default is fp16, use this option to disable it(fp32)")

    ## int8 quantization
    parser.add_argument("--int8", action="store_true", help="enable int8 quantization")

    # rknn
    parser.add_argument("--rknn", action="store_true", help="save rknn model")

    ## rknn quantization
    parser.add_argument("--rknn-platform", type=str, default="rk3588", help="rknn platform")

    # calib
    parser.add_argument("--dataset", type=str, default="cfg/dataset/coco.yaml", help="calibration dataset(int8)")
    parser.add_argument("--train", action="store_true", help="use train dataset for calibration(default: val)")
    parser.add_argument("--all", action="store_true", help="use both train and val dataset")
    parser.add_argument("--num-imgs", type=int, default=512, help="number of images for calibration, -1 for all images")

    return parser.parse_args()


@logger.catch
@torch.no_grad()
def main():

    args = get_args()

    assert any([args.onnx, args.onnx_only, args.trt, args.rknn]), "no export output!"

    if isinstance(args.input_size, int):
        args.input_size = [args.input_size] * 2
    if len(args.input_size) == 1:
        args.input_size *= 2

    exp = EdgeYOLO(weights=args.weights)
    model = exp.model

    model.fuse()
    model.eval()
    # model.cuda()

    if args.rknn:
        from edgeyolo.models.yolo import YOLOXDetect
        for k, v in model.named_modules():
            if isinstance(v, YOLOXDetect):
                v.rknn_export = True

        if args.batch > 1:
            logger.warning("Currently, RKNN export only support batch 1!, change to batch 1")
            args.batch = 1

    export_path = f"output/export/{os.path.basename(args.weights).split('.')[0]}"

    os.makedirs(export_path, exist_ok=True)

    file_name = os.path.join(export_path,
                             f"{args.input_size[0]}x{args.input_size[1]}_"
                             f"batch{args.batch}"
                             f"{'' if not args.trt else '_int8' if args.int8 else '_fp16' if not args.no_fp16 else '_fp32'}").replace("\\", "/")

    calib_dataset = None
    if args.int8:
        from edgeyolo.export import CalibDataset
        with open(args.dataset) as yamlf:
            dataset_cfg = yaml.load(yamlf, yaml.Loader)

        if args.all:
            imgs_path = [os.path.join(dataset_cfg.get("dataset_path"), dataset_cfg.get("train").get("image_dir")),
                         os.path.join(dataset_cfg.get("dataset_path"), dataset_cfg.get("val").get("image_dir"))]
        else:
            sub_dataset = "train" if args.train else "val"
            imgs_path = os.path.join(dataset_cfg.get("dataset_path"), dataset_cfg.get(sub_dataset).get("image_dir"))

        suffix = dataset_cfg.get("kwargs").get("suffix")

        calib_dataset = CalibDataset(
            dataset_path=imgs_path,
            input_size=args.input_size,
            num_image=args.num_imgs,
            pixel_range=exp.ckpt.get("pixel_range") or 255,
            suffix=suffix,
            batch=args.batch
        )
        # logger.info(calib_dataset[0][0].shape)

    x = np.ones([args.batch, 3, *args.input_size], dtype=np.float32)
    x = torch.from_numpy(x)  # .cuda()

    model(x)  # warm and init

    input_names = ["input_0"]
    output_names = ["output_0"]    # ["output_0", "output_1", "output_2"] if args.rknn else

    if args.onnx_only or args.rknn:
        if args.rknn:
            output_names = []
            for otype in ["xy", "wh", "conf"]:
                output_names.extend([f"{otype}{i}" for i in range(3)])

        import onnx
        onnx_file = file_name + ("_for_rknn" if args.rknn else "_int8" if args.int8 else "") + ".onnx"
        torch.onnx.export(model,
                          x,
                          onnx_file,
                          verbose=False,
                          opset_version=args.opset,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=None)
        onnx_model = onnx.load(onnx_file)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        if not args.no_simplify:
            try:
                import onnxsim
                logger.info('\nstart to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                logger.error(f'Simplifier failure: {e}')

        onnx.save(onnx_model, onnx_file)
        logger.info(f'ONNX export success, saved as {onnx_file}')
        data_save = {
            "names": exp.class_names,
            "img_size": args.input_size,
            "batch_size": args.batch,
            "pixel_range": exp.ckpt.get("pixel_range") or 255,  # input image pixel value range: 0-1 or 0-255
            "obj_conf_enabled": True,  # Edge-YOLO use cls conf and obj conf
            "input_name": "input_0",
            "output_name": "output_0",
            "dtype": "uint8" if args.int8 or args.rknn else "float"
        }
        with open(file_name + ".yaml", "w") as yamlf:
            yaml.dump(data_save, yamlf)

        with open(file_name + ".json", "w") as jsonf:
            json.dump(data_save, jsonf)

        if args.rknn:
            from edgeyolo.export.rknn import RKNNExporter
            rknn_file = file_name + ".rknn"
            rknn_exporter = RKNNExporter(onnx_file, rknn_file, args.rknn_platform, args.dataset, args.num_imgs, args.train, args.all)
            rknn_exporter.convert(np.ones([*args.input_size, 3], dtype="uint8"), args.batch)

    else:
        import tensorrt as trt
        from edgeyolo.export import torch2onnx2trt
        model_trt = torch2onnx2trt(
            model,
            [x],
            fp16_mode=not args.no_fp16,
            int8_mode=args.int8,
            int8_calib_dataset=calib_dataset,
            log_level=trt.Logger.INFO,
            max_workspace_size=(int((1 << 30) * args.workspace)),
            max_batch_size=args.batch,
            use_onnx=True,
            onnx_opset=args.opset,
            input_names=input_names,
            output_names=output_names,
            simplify=not args.no_simplify,
            save_onnx=file_name + ".onnx" if args.onnx else None,
            save_trt=args.trt
        )

        data_save = {
            "names": exp.class_names,
            "img_size": args.input_size,
            "batch_size": args.batch,
            "pixel_range": exp.ckpt.get("pixel_range") or 255,  # input image pixel value range: 0-1 or 0-255
            "obj_conf_enabled": True,  # Edge-YOLO use cls conf and obj conf
            "input_name": "input_0",
            "output_name": "output_0",
            "dtype": "uint8" if args.int8 else "float"
        }

        with open(file_name + ".json", "w") as jsonf:
            json.dump(data_save, jsonf)

        with open(file_name + ".yaml", "w") as yamlf:
            yaml.dump(data_save, yamlf)

        if model_trt is not None:
            data_save["model"] = model_trt.state_dict()
            torch.save(data_save, file_name + ".pt")
            logger.info("Converted TensorRT model done.")

            engine_file = file_name + ".engine"

            with open(engine_file, "wb") as f:
                f.write(model_trt.engine.serialize())

    logger.info(f"All files are saved in {export_path}.")


if __name__ == "__main__":
    main()
