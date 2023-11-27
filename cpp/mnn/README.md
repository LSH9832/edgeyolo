# MNN Deployment demo for EdgeYOLO

## 1. install MNN Framework
before using this framework, please install mnn SDK

```bash
git clone https://github.com/alibaba/MNN
cd MNN
./schema/generate.sh
mkdir build && cd build

# cpu only
cmake .. -DMNN_OPENCL=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TOOL=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_BUILD_OPENCV=ON

# cuda
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DMNN_BUILD_QUANTOOLS=ON   \
         -DMNN_BUILD_CONVERTER=ON   \
         -DMNN_OPENGL=ON            \
         -DMNN_VULKAN=ON            \
         -DMNN_CUDA=ON              \
         -DMNN_TENSORRT=OFF         \
         -DMNN_BUILD_DEMO=ON        \
         -DMNN_BUILD_BENCHMARK=ON

# if -DMNN_OPENGL=ON and openglv3 is not installed in your device, then you can run the following command
# sudo ln -s /usr/lib/x86_64-linux-gnu/libGLESv2.so /usr/lib/x86_64-linux-gnu/libGLESv3.so     # x86_64
# sudo ln -s /usr/lib/aarch64-linux-gnu/libGLESv2.so /usr/lib/aarch64-linux-gnu/libGLESv3.so   # arm64

make -j8
```

or just [**download here**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.1/mnn_dev.zip) to make sure you are using the same version with mine.
```bash
unzip mnn_dev.zip
cd mnn_dev

# if you use cpu only
chmod +x ./setup.bash && ./setup.bash

# if you want to use cuda opencl opengl
chmod +x ./setup_cuda.bash && ./setup_cuda.bash
```

## 2. setup & model preparation

- first, modify link_directories and include_directories of MNN in CMakeLists.txt
- compile this demo

```bash
chmod +x setup.bash && ./setup.bash
```
- convert your own mnn models and put it in "./models"

```bash
cd /path_to_your_MNN/build
./MNNConvert -f ONNX --modelFile XXX.onnx --MNNModel xxx.mnn --fp16
```
or you can download converted [**coco model**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.1/edgeyolo_coco.mnn) and [**visdrone model**](https://github.com/LSH9832/edgeyolo/releases/download/v1.0.1/edgeyolo_visdrone.mnn) (tiny-lrelu) and put them in dir "./models"

then modify **config/mnn_detection_xxx.yaml**:

```yaml
model: "/path_to_this_project/models/edgeyolo_coco.mnn"
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
num_threads: 3             # number of threads while doing inference
conf_thres: 0.25
nms_thres: 0.45
```


## 3. run edgeyolo with MNN

```bash
# example
./build/mnn_det --cfg config/mnn_detection_coco.yaml   \  # model config
                --video                                \  # or --device --picture
                --source /path_to_video_file.mp4       \  # or 0 for camera, or /path/to/image.jpg for picture
                --no-label                             \  # do not draw label
                --loop                                 \  # display in loop
                --cpu                                     # use cpu, or --gpu, default is auto choose
```

