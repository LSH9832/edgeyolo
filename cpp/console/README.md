# EdgeYOLO TensorRT deployment: Console demo

## note

Currently, this demo only supports models with batch = 1.

## requirements

- opencv
- qt5

## build
modify CMakeLists.txt and then
```shell
mkdir build && cd build
cmake ..
make -j4
```
## usage
Linux
```shell
# name of json and engine file should be the same.
# ./yolo [json] [source] [--options]
./yolo XXX.json /path/to/your/videos --conf 0.25 --nms 0.5 --loop --no-label
```
Windows
```shell
# name of json and engine file should be the same.
# ./yolo.exe [json] [source] [--options]
./yolo.exe XXX.json /path/to/your/videos --conf 0.25 --nms 0.5 --loop --no-label
```
