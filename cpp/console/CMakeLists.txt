cmake_minimum_required(VERSION 2.6)

project(trt_det LANGUAGES CXX)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

include_directories(
  include
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# cuda # cudnn # tensorrt
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
include_directories(/usr/local/TensorRT-8.6.1.6/include)

add_executable(trt_det
    src/yolo.cpp
    src/trt.cpp
)

target_link_libraries(trt_det nvinfer)
target_link_libraries(trt_det cudart)
target_link_libraries(trt_det yaml-cpp)
target_link_libraries(trt_det ${OpenCV_LIBS})
