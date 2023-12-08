# EdgeYOLO TensorRT deployment demo

## note

Currently, this demo only supports models with batch = 1.

## requirements

- libopencv-dev
- yaml-cpp0.6

## build
modify CMakeLists.txt and then
```shell
mkdir build && cd build
cmake ..
make
```
## usage

```shell
# name of yaml and engine file should be the same.
./yolo XXX.engine /path/to/your/videos --conf 0.25 --nms 0.5 --loop --no-label
```
