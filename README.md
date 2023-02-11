![](assets/visdrone.jpg)
# <a href="https://www.bit.edu.cn"><img src="assets/bit.png" align="left" height="60" width="60" ></a> EdgeYOLO: anchor-free, edge-friendly


## Note
This is a **trial version** without training code and evaluate code. we will publish complete source code after we submit our first edgeyolo paper.

## Intro
- In embeded device such as Nvidia Jetson AGX Xavier, EdgeYOLO reaches 34FPS with **50.6**% AP in COCO2017 dataset and **25.9**% AP in VisDrone2019 **(image input size is 640x640, batch=16, post-process included)**. And for smaller model EdgeYOLO-S, it reaches 53FPS with **44.1**% AP and **63.3**% AP<sup>0.5</sup>(**SOTA** in P5 small models) in COCO2017.
- we provide a more effective data augmentation during training.
- small object and medium object detect performace is imporved by using RH loss during the last few training epochs.

## Models & Benchmark

- models trained on COCO2017-train

|Model|Size|mAP<sup>val<br/>0.5:0.95|mAP<sup>val<br/>0.5|FPS<sup>AGX Xavier<br/>trt fp16 batch=16 <br/>include NMS|Params<br/>train / infer</br><sup>(M)|Download|
|:----|:--:|:----------------------:|:-----------------:|:-------------------------------------------------------:|:-----------------------------------:|:------:|
|**EdgeYOLO-Tiny-LRELU**|416</br>640|33.1</br>37.8|50.5</br>56.7|**206**</br>109                                  |7.6 / 7.0       |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_tiny_lrelu_coco.pth)|
|**EdgeYOLO-Tiny**|416</br>640|37.2</br>41.4|55.4</br>60.4|136</br>67                                             |5.8 / 5.5       |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_tiny_coco.pth)|
|**EdgeYOLO-S**|640|44.1            |**63.3**           |53                                                       |9.9 / 9.3       |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_s_coco.pth)|
|**EdgeYOLO-M**|640|47.5            |66.6               |46                                                       |19.0 / 17.8     |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_m_coco.pth)|
|**EdgeYOLO**|640|50.6              |69.8               |34                                                       |41.2 / 40.5     |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_coco.pth)|

- models trained on VisDrone2019 (pretrained on COCO2017-train)

| Model | Size | mAP<sup>val<br/>0.5:0.95 | mAP<sup>val<br/>0.5 |Download|
|:------|:----:|:------------------------:|:-------------------:|:------:|
|**EdgeYOLO-Tiny-LRELU**|416</br>640|12.1</br>18.5|22.8</br>33.6|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_tiny_lrelu_visdrone.pth)|
|**EdgeYOLO-Tiny**|416</br>640|14.9</br>21.8|27.3</br>38.5|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_tiny_visdrone.pth)|
|**EdgeYOLO-S**|640|23.6|40.8|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_s_visdrone.pth)|
|**EdgeYOLO-M**|640|25.0|43.0|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_m_visdrone.pth)|
|**EdgeYOLO**|640|25.9|43.9|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_visdrone.pth)|

<details>
<summary>Some of our detect results in COCO2017</summary>

![](assets/coco.jpg)

</details>

## Coming Soon
- **train code.** After our paper is released, we will publish train code.
- **evaluate code.** After our paper is released, we will publish evaluate code.

## Quick Start
### setup

```bash
git clone https://github.com/LSH9832/edgeyolo.git
cd edgeyolo
pip install -r requirements.txt
```
if you use tensorrt, please make sure torch2trt is installed
```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python setup.py install
```

### inference

**First [download weights here](https://github.com/LSH9832/edgeyolo/releases/tag/v0.0.0)**

```
python detect.py --weights edgeyolo_coco.pth --source XXX.mp4 --fp16

# full commands
python detect.py --weights edgeyolo_coco.pth 
                 --source /XX/XXX.mp4     # or dir with images, such as /dataset/coco2017/val2017    (jpg/jpeg, png, bmp, webp is available)
                 --conf-thres 0.25 
                 --nms-thres 0.5 
                 --input-size 640 640 
                 --batch-size 1 
                 --save-dir ./img/coco    # if you press "s", the current frame will be saved in this dir
                 --fp16 
                 --no-fuse                # do not fuse layers
                 --no-label               # do not draw label with class name and confidence
```
It is recomended to use **batch_detect.py** with the same commands if batch size > 1
```
python batch_detect.py --weights edgeyolo_coco.pth --source XXX.mp4 --batch-size 16 --fp16
                       --fps 30    # max fps limitation(new function)
```
### export onnx & tensorrt
```
python export_pth2onnx.py --weights edgeyolo_coco.pth --simplify

# full commands
python export_pth2onnx.py --weights edgeyolo_coco.pth 
                          --img-size 640 640 
                          --batch 1
                          --opset 11
                          --simplify
```
it generates file **yolo_export/onnx/edgeyolo_coco_640x640_batch1.onnx** and **yolo_export/onnx/edgeyolo_coco_640x640_batch1.yaml**

```
# (workspace: GB)
python export_onnx2trt.py --onnx yolo_export/onnx/edgeyolo_coco_640x640_batch1.onnx 
                          --yaml yolo_export/onnx/edgeyolo_coco_640x640_batch1.yaml 
                          --workspace 10 
                          --fp16
```

it will generate
```
yolo_export/tensorrt/edgeyolo_coco_640x640_batch1.pt         # for python inference
yolo_export/tensorrt/edgeyolo_coco_640x640_batch1.engine     # for c++ inference
yolo_export/tensorrt/edgeyolo_coco_640x640_batch1.txt        # for c++ inference
```

#### for python inference
```
python detect.py --trt --weights yolo_export/tensorrt/edgeyolo_coco_640x640.pt --source XXX.mp4

# full commands
python detect.py --trt 
                 --weights yolo_export/tensorrt/edgeyolo_coco_640x640_batch1.pt 
                 --source XXX.mp4
                 --legacy         # if "img = img / 255" when you train your train model
                 --use-decoder    # if use original yolox tensorrt model before version 0.3.0
```
It is also recomended to use **batch_detect.py** with the same commands if batch size > 1
```
python batch_detect.py --trt --weights yolo_export/tensorrt/edgeyolo_coco_640x640_batch1.pt --source XXX.mp4 --fp16
                       --fps 30    # max fps limitation(new function)
```
#### for c++ inference
it will be comming soon



