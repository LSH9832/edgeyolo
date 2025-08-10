if(WIN32)
    cmake_minimum_required(VERSION 4.1)
elseif(UNIX AND NOT APPLE)
    cmake_minimum_required(VERSION 3.10)
endif()

project(yoloinferMNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "-O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS -pthread)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)

include_directories(
    ~/envs/TensorRT-8.6.1.6/include
    ~/envs/cudnn/include
    /usr/local/cuda/include
)

link_directories(
    ~/envs/TensorRT-8.6.1.6/lib
    ~/envs/cudnn/lib
    /usr/local/cuda/lib64
)

add_library(yoloInferTensorRT SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorrt.cpp
)

target_link_libraries(yoloInferTensorRT PUBLIC
    nvinfer
    cudart
)
