# EdgeYOLO 模型端侧部署

## 说明
本人学艺不精，cmake语句不熟悉，因此交叉编译可能会有一些问题，如遇问题请自行修改，恕不回复！

## 前言

目前我们支持转换为多种推理框架的模型，如英伟达TensorRT、阿里MNN、瑞芯微RKNN、华为昇腾Ascend、地平线HBDNN(Horizon Bayes Deep Neural Network, 后续仅简称为Horizon)等。

本项目将最终支持上述所有架构模型的推理，方便您一键式部署在相应的设备上。

目前已经支持：
1. TensorRT
2. MNN
3. RKNN

**注:[cpp](../cpp)文件夹将不再进行更新**

## 特点

对应框架推理构建完成后，多个框架模型均通过一个可执行文件执行推理，无需重新构建。示例代码见[src/demo.cpp](./src/demo.cpp)，示例模型及配置文件见文件夹[models](./models)

**注：示例中tensorrt的模型是在TensorRT8.6.1.6版本下，使用RTX 3080Ti Laptop转换而来，其他非相同版本非等同算力显卡下很可能无法运行**

## 安装

在安装本项目前，请先配置好相应推理框架的环境。

进入本文件夹，修改yolo/CMakeLists.txt中对应框架头文件以及动态库路径

```bash
mkdir build && cd build

cmake -D TENSORRT=ON   \   # 构建tensorrt的yolo推理动态库，按需开启
      -D MNN=ON        \   # 构建MNN的yolo推理动态库，按需开启
      -D ROCKCHIP=ON   \   # 构建RKNN的yolo推理动态库，按需开启
      ..

make -j${nproc}

rm -rf ./*
cmake -D BUILD_API=ON  \   # 构建模型统一接口（C++）动态库，上述动态库构建完成后再开启
      -D BUILD_DEMO=ON \   # 构建示例程序
      ..
make -j${nproc}
make install
cd ..
```

**注意：仅编译一个推理动态库时，推荐按照下面的命令仅编译一次**
```bash
mkdir build && cd build
cmake -D TENSORRT=ON   \   # 仅编译tensorrt的yolo推理动态库
      -D BUILD_API=ON  \   # 构建模型统一接口（C++）动态库，上述动态库构建完成后再开启
      -D BUILD_DEMO=ON \   # 构建示例程序
      ..
make -j${nproc}
make install
cd ..
```

## 使用

此时项目安装在本项目目录下install/Linux文件夹里，根据架构分为x86_64和aarch64等文件夹，进入对应文件夹下的yolo文件夹即可。C++程序使用方式如下

```bash
# 进入对应文件夹
cd install/Linux/x86_64/yolo

# 如果因部署需要移动了本项目位置，加上下面这句
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH   

# 运行c++示例程序
bin/detect --config models/xxx.yaml  \  
           --source /path/to/your/video.mp4

# 运行python示例程序（需要安装opencv-python）
cd python
python3 detect.py --config ../models/xxx.yaml \ 
                  --source /path/to/your/video.mp4
```

其中yaml配置文件应至少包含如下信息

```yaml
model_path: ./bdd100k.mnn  # 模型文件相对于本配置文件的位置
batch_size: 1              # 目前只支持batch size=1, 如需批量推理请自行修改源码
img_size: [384, 640]       # 输入图像高、宽
input_name: input_0        # 输入名称，仅支持一个输入
names:                     # 类别名称，这里展示的是bdd100k数据集的类别名称
- person
- rider
- car
- bus
- truck
- bike
- motor
- tl_green
- tl_red
- tl_yellow
- tl_none
- traffic_sign
- train
obj_conf_enabled: true    # 是否使用了前景置信度，不需要管，目前来说只要不是yolov6的模型就设置为true
output_names:             # 输出张量名称列表
- output_0
normalize: false          # 输入是否需要归一化

kwargs:
  conf_threshold: 0.75    # 置信度阈值
  nms_threshold: 0.5      # NMS的IOU阈值
```

如果要自己使用统一yolo接口，则在自己的项目CMakeLists.txt加上以下语句(假设路径就在这个项目下)

```cmake
include_directories(
    ./install/Linux/x86_64/yolo/include
)

link_directories(
    ./install/Linux/x86_64/yolo/lib
)

add_executable(YOUR_PROJECT_NAME
    ...
)

target_link_libraries(YOUR_PROJECT_NAME
    YOLODetect
)
```

使用方式也很简单，如下

```c++
#include "detect.hpp"


int main()
{
    // 初始化检测器
    Detector detector;
    detector.init("path/to/config.yaml")

    // 待检测图像
    cv::Mat image = cv::imread("path/to/img.jpg");

    // 检测目标信息
    std::vector<std::vector<float>> objects;
    detector.detect(image, results);

    // 解析信息
    for(auto& object: objects)
    {
        // 目标信息
        int x1=static_cast<int>(object[0]);
        int y1=static_cast<int>(object[1]);
        int x2=static_cast<int>(object[2]);
        int y2=static_cast<int>(object[3]);
        int label=static_cast<int>(object[4]);
        float confidence = object[5];
        // do something

    }

    // 结果可视化
    cv::imshow("yolo result visualize", detector.draw(image, results));
    cv::waitKey(0);

    return 0
}

```
