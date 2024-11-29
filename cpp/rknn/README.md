# EdgeYOLO RKNN deployment tool

## note: this rknn code can only run in linux and only for rk3588.

| Model |   Size    | FPS<sup>RK3588, 3 cores<br/>int8 batch=1 <br/>include all process |                                                                                                     Download                                                                                                      |
|:------|:---------:|:-----------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|**Tiny-LRELU**|  384×640  |                                65                                 | [**coco model**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.2/edgeyolo_tiny_lrelu_coco.rknn), [**coco config**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.2/edgeyolo_tiny_lrelu_coco.yaml) </br> [**visdrone model**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.2/edgeyolo_tiny_lrelu_visdrone.rknn), [**visdrone config**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.2/edgeyolo_tiny_lrelu_visdrone.yaml) |
|**Tiny**      |  384×640  |              24 </br>(too many SiLU activate layers)              |       [**coco model**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.2/edgeyolo_tiny_coco.rknn), [**coco config**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.2/edgeyolo_tiny_coco.yaml) </br> [**visdrone model**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.2/edgeyolo_tiny_visdrone.rknn), [**visdrone config**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.2/edgeyolo_tiny_visdrone.yaml)      |

## 1. Model convert
### 1.1 Setup RKNN Toolkit2 on your PC(x86_64) with edgeyolo requirements

- this toolkit only supports python3.6, python3.8 and python3.10 

```bash
cd toolkit_install
./rknn_toolkit_install 38   # or 36, or 310
```

### 1.2 convert your model

- or just use the rknn models given in the above table.

```bash
cd /path_to_edgeyolo_project_root_path

python export.py --weights edgeyolo_tiny_lrelu_coco.pth \   # your pth weights
                 --input-size 384 640 \  # for 16:9, if 4:3, use 480 640
                 --rknn               \  # export rknn model
                 --dataset cfg/dataset/coco.yaml \  # calib dataset
                 --num-img 100        \  # number of calib img
                 
                 # optional but not commend
                 --rknn-platform rk3588 \  # rk3566 and so on, you can convert model, but our code only support rk3588(and rk3588s)
```

then it generates 4 files as follows
```
output/export/edgeyolo_tiny_lrelu_coco/384x640_batch1.rknn             # rknn model
output/export/edgeyolo_tiny_lrelu_coco/384x640_batch1_for_rknn.onnx    # different from onnx file for tensorrt
output/export/edgeyolo_tiny_lrelu_coco/384x640_batch1.json             # json file, not used currently
output/export/edgeyolo_tiny_lrelu_coco/384x640_batch1.yaml             # config file to use
```

## 2. Run rknn model in your rk3588 device(arm64)

if you want to use the newest rknn library, see [official website](https://github.com/airockchip/rknn-toolkit2)

- copy dir '**cpp/rknn**' to your rk3588 device.
- cd rknn
- copy converted ".yaml" and ".rknn" file to ./model. if rename, rename both file with the same name.
- then
```bash
chmod +x ./setup_rk3588.sh
./setup_rk3588.sh

cd install/rknn_edgeyolo_demo_Linux

./rknn_det -?   # parser helper

./rknn_det --model model/384x640_batch1.rknn \
           --video \   # use video source(include rtsp/rtmp), or --device for camera id, or --picture for single image.
           --source /path_to_your_video.mp4 \   # or 0, or /path_to_your_image.jpg
           --no-label \   # draw bounding box without label
           --loop   # play in loop, press esc to quit
```
