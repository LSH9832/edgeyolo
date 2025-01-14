import os
import json
import yaml
import torch
import argparse
import numpy as np
import os.path as osp

from loguru import logger

from edgeyolo import EdgeYOLO


def get_args():
    parser = argparse.ArgumentParser("EdgeYOLO Docker Export Parser")

    # basic
    parser.add_argument("--weights", type=str, default="./weights/edgeyolo_tiny_coco.pth")
    parser.add_argument("--input-size", type=int, nargs="+", default=[640, 640])
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')

    # onnx
    parser.add_argument("--onnx", action="store_true", help="save onnx model(if tensorrt and torch2trt are installed)")
    parser.add_argument("--onnx-only", action="store_true", help="(if tensorrt and torch2trt are not installed)")
    parser.add_argument("--no-simplify", action="store_true", help="do not simplify models(not recommend)")
    parser.add_argument("--force-decode", action="store_true", help="force add decoder to network")
    
    parser.add_argument("--opset", type=int, default=11, help="onnx opset")

    # parser.add_argument("--relu", action="store_true", help="replace silu with relu")

    # tensorrt
    parser.add_argument("--trt", action="store_true", help="save tensorrt models")
    parser.add_argument("-w", '--workspace', type=float, default=8, help='max workspace size(GB)')

    ## fp16 quantization
    parser.add_argument("--no-fp16", action="store_true", help="default is fp16, use this option to disable it(fp32)")

    ## int8 quantization
    parser.add_argument("--int8", action="store_true", help="enable int8 quantization")

    # mnn
    parser.add_argument("--mnn", action="store_true", help="save mnn model")

    # rknn
    parser.add_argument("--rknn", action="store_true", help="save rknn model")

    # horizon j5
    parser.add_argument("--j5", action="store_true", help="save bin model file for horizon j5 platform")

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

    assert any([args.onnx, args.onnx_only, args.trt, args.rknn, args.mnn, args.j5]), "no export output!"

    if isinstance(args.input_size, int):
        args.input_size = [args.input_size] * 2
    if len(args.input_size) == 1:
        args.input_size *= 2

    exp = EdgeYOLO(weights=args.weights)
    model = exp.model
    # if not args.mnn:
    model.fuse()
    model.eval()
    # model.cuda()

    if (args.rknn or args.int8 or args.j5) and not args.force_decode:
        from edgeyolo.models.yolo import YOLOXDetect
        for _, v in model.named_modules():
            if isinstance(v, YOLOXDetect):
                v.int8_export = True

        if args.rknn and args.batch > 1:
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

    result = model(x)  # warm and init
    # print(result)

    input_names = ["input_0"]
    output_names = ["output_0"]

    if args.onnx_only or args.rknn or args.mnn or args.j5:
        if (args.rknn or args.int8 or args.j5) and not args.force_decode:
            output_names = []
            for otype in ["reg", "obj_conf", "cls_conf"]:
                output_names.extend([f"{otype}{i}" for i in range(3)])

        import onnx
        file_name += "_for_rknn" if args.rknn else "_for_j5" if args.j5 \
                     else "_for_mnn" if args.mnn else "_int8" if args.int8 else ""
        onnx_file = file_name + ".onnx"
        
        torch.onnx.export(model,
                          x,
                          onnx_file,
                          verbose=False,
                          opset_version=12 if args.rknn else args.opset,
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
        if args.mnn:
            data_save = {
                "model": file_name + ".mnn",
                "names": exp.class_names,
                "num_threads": min(7, max(3, os.cpu_count()-3)),  # number of threads while doing inference
                "conf_thres": 0.25,
                "nms_thres": 0.45
            }
        else:
            data_save = {
                "names": exp.class_names,
                "img_size": args.input_size,
                "batch_size": args.batch,
                "pixel_range": exp.ckpt.get("pixel_range") or 255,  # input image pixel value range: 0-1 or 0-255
                "obj_conf_enabled": True,  # Edge-YOLO use cls conf and obj conf
                "input_name": input_names[0],
                "output_name": output_names if len(output_names) > 1 else output_names[0],
                "dtype": "uint8" if args.int8 or args.rknn else "float"
            }
        with open(file_name + ".yaml", "w") as yamlf:
            yaml.dump(data_save, yamlf)

        with open(file_name + ".json", "w") as jsonf:
            json.dump(data_save, jsonf)
        
        if args.mnn:
            mnn_fp16_file = file_name + ".mnn"
            command = f"MNNConvert -f ONNX --modelFile {onnx_file} --MNNModel {mnn_fp16_file} --fp16"
            
            os.system(command)
            if args.int8:
                mnn_int8_file = file_name + "_int8.mnn"
                calib_image_dir, json_config = gen_calib_data_mnn(
                    dist_path=export_path,
                    dataset=args.dataset,
                    input_size=args.input_size,
                    num_imgs=args.num_imgs
                )
                calib_command = f"~/code/MNN-2.7.1/build/quantized.out {osp.abspath(mnn_fp16_file)} {osp.abspath(mnn_int8_file)} {osp.abspath(json_config)}"
                logger.info(f"now do calib:\n{calib_command}")
                os.system(calib_command)
                
        
        elif args.j5:
            export_yaml_file = file_name + "_export_params.yaml"
            with open(export_yaml_file, "w") as yamlf:
                
                yaml.dump(horizon_params(
                    onnx_file=onnx_file,
                    dist_path=export_path,
                    file_name=file_name,
                    calib_data_path=gen_calib_data_for_horizon_platform(
                        dist_path=export_path,
                        dataset=args.dataset,
                        input_size=args.input_size,
                        num_imgs=args.num_imgs
                    ), 
                    input_size=args.input_size,
                    batch=args.batch,
                    j5=args.j5
                ), yamlf)
            command = f"hb_mapper makertbin --config {export_yaml_file} --model-type onnx"
            os.system(command)

        elif args.rknn:
            from edgeyolo.export.rknn import RKNNExporter
            rknn_file = file_name + ".rknn"
            rknn_exporter = RKNNExporter(onnx_file, rknn_file, args.rknn_platform, args.dataset, args.num_imgs, args.train, args.all)
            rknn_exporter.convert(np.ones([*args.input_size, 3], dtype="uint8"), args.batch)

    else:
        import tensorrt as trt
        from edgeyolo.export import torch2onnx2trt

        if args.int8 and not args.force_decode:
            output_names = []
            for otype in ["reg", "obj_conf", "cls_conf"]:
                output_names.extend([f"{otype}{i}" for i in range(3)])
        
        data_save = {
            "names": exp.class_names,
            "img_size": args.input_size,
            "batch_size": args.batch,
            "pixel_range": exp.ckpt.get("pixel_range", 255),  # input image pixel value range: 0-1 or 0-255
            "obj_conf_enabled": True,  # Edge-YOLO use cls conf and obj conf
            "input_name": "input_0",
            "output_name": output_names,
            "dtype": "uint8" if args.int8 else "float"
        }

        with open(file_name + ".json", "w") as jsonf:
            json.dump(data_save, jsonf)

        with open(file_name + ".yaml", "w") as yamlf:
            yaml.dump(data_save, yamlf)
        
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

        

        if model_trt is not None:
            data_save["model"] = model_trt.state_dict()
            torch.save(data_save, file_name + ".pt")
            logger.info("Converted TensorRT model done.")

            engine_file = file_name + ".engine"

            with open(engine_file, "wb") as f:
                f.write(model_trt.engine.serialize())

    logger.info(f"All files are saved in {export_path}.")


def gen_calib_data_for_horizon_platform(dist_path, dataset, input_size, num_imgs=50):
    from glob import glob
    from random import shuffle
    import cv2
    from edgeyolo.data.data_augment import preproc

    # os.makedirs(dist_path, exist_ok=True)

    dataset_cfg: dict = yaml.load(open(dataset), yaml.Loader)
    dataset_path = dataset_cfg.get("dataset_path")
    img_dir = dataset_cfg.get("val").get("image_dir")
    img_path = osp.join(dataset_path, img_dir)
    suffix = dataset_cfg.get('kwargs').get('suffix')
    img_files = glob(osp.join(img_path, f"*.{suffix}"))
    shuffle(img_files)
    # img_files = img_files[:num_imgs]
    dist_dir = osp.join(dist_path, "horizon_calib_data", f"{input_size[0]}x{input_size[1]}")
    os.makedirs(dist_dir, exist_ok=True)

    img_type = np.float32

    current_num = 0
    bgr_files = []
    for efile in glob(osp.join(dist_dir, "*")):
        if osp.isdir(efile):
            logger.warning(f"please move dir:{efile} to other path.")
        elif not efile.endswith(".bgr"):
            logger.warning(f"{efile} is not the file for calibration, please move it to other path.")
        else:
            current_num += 1
            bgr_files.append(efile)
    
    if current_num >= num_imgs:
        shuffle(bgr_files)
        for file_to_remove in bgr_files[:current_num - num_imgs]:
            print(f"remove data file {file_to_remove}")
            os.remove(file_to_remove)
        return dist_dir

    
    for img_file in img_files:
        im_name = osp.basename(img_file)[:-len(suffix)-1]
        im_dist = osp.join(dist_dir, f"{im_name}.bgr")

        if osp.isfile(im_dist):
            continue

        im_ori = cv2.imread(img_file)
        
        # print(im_name)

        im_, _ = preproc(im_ori, input_size)
        im_save = np.array([im_]).astype(img_type)
        
        # print(np.shape(im_save))

        im_save.tofile(im_dist)
        if osp.isfile(im_dist):
            print(f"calib data saved to {im_dist}")
            current_num += 1
        else:
            print(f"failed to save data to {im_dist}")
        if current_num == num_imgs:
            break
    print("CALIB DATA GENERATION DONE")
    return dist_dir


def horizon_params(onnx_file, dist_path, file_name, calib_data_path, 
                   input_size, batch=1, j5=True):
    return {
        # 模型参数组
        "model_parameters": {
            "onnx_model": osp.abspath(onnx_file),  # 原始Onnx浮点模型文件
            "march": 'bayes' if j5 else "bernoulli2", # 转换的目标处理器架构        
            "output_model_file_prefix": os.path.basename(file_name), # 模型转换输出的用于上板执行的模型文件的名称前缀
            "working_dir": osp.abspath(dist_path),  # 模型转换输出的结果的存放目录
            "layer_out_dump": False  # 指定转换后混合异构模型是否保留输出各层的中间结果的能力
        }, 

        # 输入信息参数组
        "input_parameters": {
            "input_name": "input_0",        # 原始浮点模型的输入节点名称
            "input_type_train": 'bgr',      # 原始浮点模型的输入数据格式（数量/顺序与input_name一致）
            "input_layout_train": 'NCHW',  # 原始浮点模型的输入数据排布（数量/顺序与input_name一致）
            "input_shape": f'1x3x{input_size[0]}x{input_size[1]}',  # 原始浮点模型的输入数据尺寸
            "input_batch": batch,  # 网络实际执行时，输入给网络的batch_size, 默认值为1
            "norm_type": 'no_preprocess',   # 在模型中添加的输入数据预处理方法
            "input_type_rt": 'bgr',  # 转换后混合异构模型需要适配的输入数据格式（数量/顺序与input_name一致）
            "input_layout_rt" : 'NHWC',
            "input_space_and_range": 'regular',  # 输入数据格式的特殊制式
        }, 

        # 校准参数组
        "calibration_parameters": {
            "cal_data_dir": osp.abspath(calib_data_path),  # 模型校准使用的标定样本的存放目录
            "cal_data_type": 'float32', # 指定校准数据二进制文件的数据存储类型。
            "calibration_type": 'mix',  # 'kl',  # 校准使用的算法类型
            "max_percentile": 1.0,  # max 校准方式的参数
            "per_channel": False  # 指定是否针对每个channel进行校准
        },
        
        # 编译参数组
        "compiler_parameters": {
            "compile_mode": 'latency',  # 编译策略选择
            "debug": True,  # 是否打开编译的debug信息
            "core_num": 1,  # 模型运行核心数
            "optimize_level": 'O3',  # 模型编译的优化等级选择
            "max_time_per_fc": 0,   # 指定模型的每个function call的最大可连续执行时间
            "jobs": os.cpu_count()      # 指定编译模型时的进程数
        }
    }


def gen_calib_data_mnn(dist_path, dataset, input_size, num_imgs=64):
    from glob import glob
    from random import shuffle
    import cv2
    from edgeyolo.data.data_augment import preproc

    dataset_cfg: dict = yaml.load(open(dataset), yaml.Loader)
    dataset_path = dataset_cfg.get("dataset_path")
    img_dir = dataset_cfg.get("val").get("image_dir")
    img_path = osp.join(dataset_path, img_dir)
    suffix = dataset_cfg.get('kwargs').get('suffix')
    img_files = glob(osp.join(img_path, f"*.{suffix}"))
    shuffle(img_files)
    # img_files = img_files[:num_imgs]
    dist_dir = osp.join(dist_path, "mnn_calib_data", f"{input_size[0]}x{input_size[1]}")
    os.makedirs(dist_dir, exist_ok=True)

    img_type = np.uint8

    current_num = 0
    bgr_files = []
    for efile in glob(osp.join(dist_dir, "*")):
        if osp.isdir(efile):
            logger.warning(f"please move dir:{efile} to other path.")
        elif not efile.endswith(".jpg"):
            logger.warning(f"{efile} is not the file for calibration, please move it to other path.")
        else:
            current_num += 1
            bgr_files.append(efile)
    
    if current_num >= num_imgs:
        shuffle(bgr_files)
        for file_to_remove in bgr_files[:current_num - num_imgs]:
            print(f"remove data file {file_to_remove}")
            os.remove(file_to_remove)
    else:
        for img_file in img_files:
            im_name = osp.basename(img_file)[:-len(suffix)-1]
            im_dist = osp.join(dist_dir, f"{im_name}.jpg")

            if osp.isfile(im_dist):
                continue

            im_ori = cv2.imread(img_file)
            
            # print(im_name)

            im_, _ = preproc(im_ori, input_size, (0, 1, 2))
            im_save = im_.astype(img_type)
            
            # print(np.shape(im_save))

            # im_save.tofile(im_dist)
            cv2.imwrite(im_dist, im_save)
            if osp.isfile(im_dist):
                print(f"calib data saved to {im_dist}")
                current_num += 1
            else:
                print(f"failed to save data to {im_dist}")
            if current_num == num_imgs:
                break
    
    mnn_calib_params = {
        "format": "BGR",
        "width": input_size[1],
        "height": input_size[0],
        "path": dist_dir,
        "used_sample_num": num_imgs,
        "feature_quantize_method": "KL",
        "weight_quantize_method": "MAX_ABS",
        "batch_size": 1,
        "debug": True
    }
    print(mnn_calib_params)
    
    json_file = osp.join(dist_path, "mnn_calib_data", f"{input_size[0]}x{input_size[1]}.json")
    with open(json_file, "w") as f:
        json.dump(mnn_calib_params, f)
    
    assert osp.isfile(json_file), "can not save json file " + json_file
    
    print("CALIB DATA GENERATION DONE")
    return osp.abspath(dist_dir), osp.abspath(json_file)


if __name__ == "__main__":
    main()
