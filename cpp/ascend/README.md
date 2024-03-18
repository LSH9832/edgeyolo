# EdgeYOLO deployment for Huawei Ascent devices

## requirements

- 1. Ascend-cann-toolkit_X.X.X_linux-xxxxx.run from Huawei Official Website.
- 2. Ascend-cann-amct_X.X.X_linux-xxxxx.zip from Huawei Official Website.
- 3. other common requiements

## model convertion
```bash
# firstly use export.py/docker_export.py to export your onnx 
# then use convertion tools, need requirements 2
python demo/amct_onnx2om.py -i /path/to/your/onnx/file.onnx \ 
                            -b 16 \
                            --dataset cfg/dataset/coco.yaml \  # select calib images from val set of this dataset
                            --om \  # convert to om file, need requirements 1
                            --fusion   
```

## run

### compile

```bash
mkdir build && cd build
cmake ..
make -j4
```

### prepare

- generate yaml file like this
```yaml
# cfg/model.yaml
model_path: /absolute/path/to/your/model.om
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
```

- finally, run this demo
```bash
./ascend_det --cfg cfg/model.yaml --source /path/to/your/video.mp4
```
