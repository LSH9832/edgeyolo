![](assets/visdrone.jpg)
# <a href="https://www.bit.edu.cn"><img src="assets/bit.png" align="left" height="60" width="60" ></a> EdgeYOLO: anchor-free, edge-friendly

<div align="center">

[简体中文](README_CN.md)

</div>


**[1 Intro](#intro)**</br>
**[2 Coming Soon](#coming-soon)**</br>
**[3 Models](#models)**</br>
**[4 Quick Start](#quick-start)**</br>
$\quad$[4.1 setup](#setup)</br>
$\quad$[4.2 inference](#inference)</br>
$\quad$[4.3 train](#train)</br>
$\quad$[4.4 evaluate](#evaluate)</br>
$\quad$[4.5 export onnx & tensorrt](#export-onnx--tensorrt)</br>
**[5 Cite EdgeYOLO](#cite-edgeyolo)**</br>
**[6 Bugs found currently](#bugs-found-currently)**

## Intro
- In embeded device such as Nvidia Jetson AGX Xavier, EdgeYOLO reaches 34FPS with **50.6**% AP in COCO2017 dataset and **25.9**% AP in VisDrone2019 **(image input size is 640x640, batch=16, post-process included)**. And for smaller model EdgeYOLO-S, it reaches 53FPS with **44.1**% AP and **63.3**% AP<sup>0.5</sup>(**SOTA** in P5 small models) in COCO2017.
- we provide a more effective data augmentation during training.
- small object and medium object detect performace is imporved by using RH loss during the last few training epochs.
- Our pre-print paper is released on [**arxiv**](https://arxiv.org/abs/2302.07483).

## Coming Soon
- TensorRT int8 export code with **Calibration** (**torch2trt** is required)
- MNN deployment code
- More different models
- C++ code for TensorRT inference
- EdgeYOLO-mask for segmentation task
- Simple but effective pretrain method

## Models

- models trained on COCO2017-train

| Model | Size | mAP<sup>val<br/>0.5:0.95 | mAP<sup>val<br/>0.5 | FPS<sup>AGX Xavier<br/>trt fp16 batch=16 <br/>include NMS | Params<br/>train / infer</br><sup>(M) |Download|
|:------|:----:|:------------------------:|:-------------------:|:---------------------------------------------------------:|:-------------------------------------:|:------:|
|**EdgeYOLO-Tiny-LRELU**|416</br>640|33.1</br>37.8|50.5</br>56.7|**206**</br>109|7.6 / 7.0  |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_tiny_lrelu_coco.pth)|
|**EdgeYOLO-Tiny**      |416</br>640|37.2</br>41.4|55.4</br>60.4|136</br>67     |5.8 / 5.5  |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_tiny_coco.pth)|
|**EdgeYOLO-S**         |640        |44.1         |**63.3**     |53             |9.9 / 9.3  |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_s_coco.pth)|
|**EdgeYOLO-M**         |640        |47.5         |66.6         |46             |19.0 / 17.8|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_m_coco.pth)|
|**EdgeYOLO**           |640        |50.6         |69.8         |34             |41.2 / 40.5|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_coco.pth)|

- models trained on VisDrone2019 (pretrained backbone on COCO2017-train)

we use [ VisDrone2019-DET dataset with COCO format ](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.0/visdrone_coco.zip) in our training.

| Model | Size | mAP<sup>val<br/>0.5:0.95 | mAP<sup>val<br/>0.5 |Download|
|:------|:----:|:------------------------:|:-------------------:|:------:|
|**EdgeYOLO-Tiny-LRELU**|416</br>640|12.1</br>18.5|22.8</br>33.6|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_tiny_lrelu_visdrone.pth)|
|**EdgeYOLO-Tiny**      |416</br>640|14.9</br>21.8|27.3</br>38.5|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_tiny_visdrone.pth)|
|**EdgeYOLO-S**         |640        |23.6         |40.8         |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_s_visdrone.pth)|
|**EdgeYOLO-M**         |640        |25.0         |42.9         |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_m_visdrone.pth)|
|**EdgeYOLO**           |640        |25.9</br>26.4|43.9</br>44.8|[**github(legacy)**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_visdrone.pth)</br>[**github(new)**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.0/edgeyolo_visdrone.pth)|

<details>
<summary>Some of our detect results in COCO2017</summary>

COCO2017
![](assets/coco.jpg)

</details>



## Quick Start
### setup

```shell
git clone https://github.com/LSH9832/edgeyolo.git
cd edgeyolo
pip install -r requirements.txt
```

if you use tensorrt, please make sure torch2trt and TensorRT Development Toolkit(version>7.1.3.0) is installed.

```shell
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python setup.py install
```
or to make sure you use the same version of torch2trt as ours, [download here](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.0/torch2trt.zip)

### inference

**First [download weights here](https://github.com/LSH9832/edgeyolo/releases/tag/v0.0.0)**

```shell
python detect.py --weights edgeyolo_coco.pth --source XXX.mp4 --fp16

# full commands
python detect.py --weights edgeyolo_coco.pth 
                 --source /XX/XXX.mp4     # or dir with images, such as /dataset/coco2017/val2017    (jpg/jpeg, png, bmp, webp is available)
                 --conf-thres 0.25 
                 --nms-thres 0.5 
                 --input-size 640 640 
                 --batch 1 
                 --save-dir ./output/detect/imgs    # if you press "s", the current frame will be saved in this dir
                 --fp16 
                 --no-fuse                # do not reparameterize model
                 --no-label               # do not draw label with class name and confidence
```

It is recomended to use **batch_detect.py** with the same commands if batch size > 1
```shell
python batch_detect.py --weights edgeyolo_coco.pth --source XXX.mp4 --batch 16 --fp16
                       --fps 30    # max fps limitation(new function)
```

### train
- first preparing your dataset and create dataset config file(./params/dataset/XXX.yaml), make sure your dataset config file contains:

(COCO, VOC, VisDrone and DOTA formats are supported)
```yaml
type: "coco"                        # dataset format(lowercase)，COCO, VOC, VisDrone and DOTA formats are supported currently
dataset_path: "/dataset/coco2017"   # root dir of your dataset

kwargs:
  suffix: "jpg"        # suffix of your dataset's images
  use_cache: true      # test on i5-12490f: Total loading time: 52s -> 10s(seg enabled) and 39s -> 4s(seg disabled)

train:
  image_dir: "images/train2017"                   # train set image dir
  label: "annotations/instances_train2017.json"   # train set label file(format with single label file) or directory(multi label files)

val:
  image_dir: "images/val2017"                     # evaluate set image dir
  label: "annotations/instances_val2017.json"     # evaluate set label file or directory

test:
  test_dir: "test2017"     # test set image dir (not used in code now, but will)

segmentaion_enabled: true  # whether this dataset has segmentation labels and you are going to use them instead of bbox labels

names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']    # category names
```
- then edit file ./params/train/train_XXX.yaml
- finally
```shell
python train.py --cfg ./params/train/train_XXX.yaml
```

### evaluate
```shell
python evaluate.py --weights edgeyolo_coco.pth --dataset params/dataset/XXX.yaml --batch 16 --device 0

# full commands
python evaluate.py --weights edgeyolo_coco.pth 
                   --dataset params/dataset/XXX.yaml 
                   --batch 16   # batch size for each gpu
                   --device 0
                   --input-size 640 640   # height, width
```

### export onnx & tensorrt
```shell
python export_pth2onnx.py --weights edgeyolo_coco.pth --simplify

# full commands
python export_pth2onnx.py --weights edgeyolo_coco.pth 
                          --input-size 640 640   # height, width
                          --batch 1
                          --opset 11
                          --simplify
```
it generates 2 files 
- **output/export/onnx/edgeyolo_coco_640x640_batch1.onnx**
- **output/export/onnx/edgeyolo_coco_640x640_batch1.yaml** for TensorRT conversion.

```shell
# (workspace: GB)
python export_onnx2trt.py --onnx output/export/onnx/edgeyolo_coco_640x640_batch1.onnx 
                          --yaml output/export/onnx/edgeyolo_coco_640x640_batch1.yaml # if name is the same as onnx file, you can skip writing this line
                          --workspace 10 
                          --fp16   # --int8 and --best: calibration training needed
```
it will generate
- **output/export/tensorrt/edgeyolo_coco_640x640_batch1.pt**  for python inference
- **output/export/tensorrt/edgeyolo_coco_640x640_batch1.engine**  for c++ inference
- **output/export/tensorrt/edgeyolo_coco_640x640_batch1.txt**  for c++ inference
- **output/export/tensorrt/edgeyolo_coco_640x640_batch1.json**  for c++ QT inference

#### for python inference
```shell
python detect.py --trt --weights output/export/tensorrt/edgeyolo_coco_640x640_batch1.pt --source XXX.mp4

# full commands
python detect.py --trt 
                 --weights output/export/tensorrt/edgeyolo_coco_640x640_batch1.pt 
                 --source XXX.mp4
                 --legacy         # if "img = img / 255" when you train your train model
                 --use-decoder    # if use original yolox tensorrt model before version 0.3.0
```

It is also recomended to use **batch_detect.py** with the same commands if batch size > 1

```shell
python batch_detect.py --trt --weights output/export/tensorrt/edgeyolo_coco_640x640_batch1.pt --source XXX.mp4 --fp16
                       --fps 30    # max fps limitation(new function)
```

#### for c++ inference
it will be coming soon


## Cite EdgeYOLO
```
 @article{edgeyolo2023,
  title={EdgeYOLO: An Edge-Real-Time Object Detector},
  author={Shihan Liu, Junlin Zha, Jian Sun, Zhuo Li, and Gang Wang},
  journal={arXiv preprint arXiv:2302.07483},
  year={2023}
}
```

## Bugs found currently
- Sometimes it raises error as follows during training. Reduce pytorch version to 1.8.0 might solve this problem.
```
File "XXX/edgeyolo/edgeyolo/train/loss.py", line 667, in dynamic_k_matching
_, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```
- For DOTA dataset, we only support single GPU training mode now, please do not train DOTA dataset with distributed mode or model can not be trained correctly.
- Sometimes converting to TensorRT fp16 model with 8.4.X.X or higher version might lose a lot of precision, please use TensorRT Verson 7.X.X.X or 8.2.X.X
