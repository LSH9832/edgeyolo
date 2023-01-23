# EdgeYOLO: anchor-free, edge-computing-device-friendly

![](assets/visdrone_result.jpg)

## Note
This is a **trial version** without training code and evaluate code. we will publish complete source code after we publish our first edgeyolo paper.

## Intro
- In embeded device such as Nvidia Jetson AGX Xavier, EdgeYOLO reaches 34FPS with **50.6**% AP in COCO2017 dataset and **25.9**% AP in VisDrone2019 **(image input size is 640x640, batch=16, post-process included)**. And for smaller model EdgeYOLO-S, it reaches 50FPS with **44.1**% AP and **63.3**% AP<sup>0.5</sup>(**SOTA** in P5 small models) in COCO2017.
- we provide a more effective data augmentation during training.
- small object and medium object detect performace is imporved by using RH loss during the last few training epochs.

## Models

- models trained on COCO2017-train

Now we are traning the model **EdgeYOLO-Tiny** and it will be comming in no more than 3 days.

| Model           | Size      | mAP<sup>val<br/>0.5:0.95 | mAP<sup>val<br/>0.5 | FPS<sup>AGX Xavier<br/>trt fp16 batch=16 <br/>include NMS | Params<br/>train / infer</br>(M) |Download|
| :-------------- | --------- | :----------------------- | ------------------- | ------------------------------------------------------------------ | -------------- |--------|
|**EdgeYOLO-Tiny (Epoch 231/300)**|416</br>640|-</br>38.0             |-</br>56.8        |-</br>-                                                          |5.8 / 5.5       |-|
|**EdgeYOLO-Tiny-LRELU**|416</br>640|33.1</br>37.8             |50.5</br>56.7        |**206**</br>96                                                          |7.6 / 7.0       |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_tiny_lrelu_coco.pth)|
|**EdgeYOLO-S**   |640        |44.1                      |**63.3**             |50                                                                  |9.9 / 9.3       |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_s_coco.pth)|
|**EdgeYOLO**     |640        |50.6                      |69.7                 |34                                                                  |41.2 / 40.5     |[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_coco.pth)|

- models trained on VisDrone2019 (pretrained on COCO2017-train)

| Model | Size | mAP<sup>val<br/>0.5:0.95 | mAP<sup>val<br/>0.5 |Download|
| :---- | ---- | :----------------------- | ------------------- |--------|
|**EdgeYOLO-Tiny (coming soon)**|416</br>640|-</br>-|-</br>-|**-**|
|**EdgeYOLO-Tiny-LRELU**|416</br>640|12.1</br>18.5|22.8</br>33.6|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_tiny_lrelu_visdrone.pth)|
|**EdgeYOLO-S**|640|23.6|40.8|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_s_visdrone.pth)|
|**EdgeYOLO**|640|25.9|43.9|[**github**](https://github.com/LSH9832/edgeyolo/releases/download/v0.0.0/edgeyolo_visdrone.pth)|

<details>
<summary>Some of our detect results in COCO2017</summary>

![](assets/coco_result.jpg)

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
python pth2onnx.py --weights edgeyolo_coco.pth --simplify

# full commands
python pth2onnx.py --weights edgeyolo_coco.pth 
                   --img-size 640 640 
                   --batch-size 1
                   --opset 11
                   --simplify
```
it generates file **export_output/onnx/edgeyolo_coco_640x640_batch1.onnx** and **export_output/onnx/edgeyolo_coco_640x640_batch1.yaml**

```
# (workspace: GB)
python onnx2trt.py --onnx export_output/onnx/edgeyolo_coco_640x640_batch1.onnx 
                   --yaml export_output/onnx/edgeyolo_coco_640x640_batch1.yaml 
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
python detect.py --trt --weights export_output/tensorrt/edgeyolo_coco_640x640.pt --source XXX.mp4

# full commands
python detect.py --trt 
                 --weights export_output/tensorrt/edgeyolo_coco_640x640_batch1.pt 
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



