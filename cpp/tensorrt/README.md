# EdgeYOLO TensorRT deployment demo

## note

Currently, this demo only supports models with batch = 1.

## requirements

- libopencv-dev
- libyaml-cpp0.6

## build
modify CMakeLists.txt and then
```shell
mkdir build && cd build
cmake ..
make
```
## usage

copy your engine file and yaml file to a same path, then

```shell
# name of yaml and engine file should be the same.
./trt_det /path/to/your/model.engine /path/to/your/videos --conf 0.25 --nms 0.5 --loop --no-label
```
