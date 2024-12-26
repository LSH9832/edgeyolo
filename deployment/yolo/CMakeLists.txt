cmake_minimum_required(VERSION 3.0.2)

project(yoloInfer LANGUAGES CXX)

add_definitions(-std=c++14)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS -pthread)
set(CMAKE_BUILD_TYPE Release)


message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/lib/")

if (TENSORRT)
    option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    find_package(CUDA REQUIRED)

    include_directories(~/envs/TensorRT-8.6.1.6/include)
    include_directories(~/envs/cudnn/include)
    include_directories(/usr/local/cuda/include)

    link_directories(~/envs/TensorRT-8.6.1.6/lib)
    link_directories(~/envs/cudnn/lib)
    link_directories(/usr/local/cuda/lib64)
    

    add_library(yoloInferTensorRT SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/platform/tensorrt/tensorrt.cpp
    )

    target_link_libraries(yoloInferTensorRT PUBLIC
        nvinfer
        cudart
    )
endif()

if (MNN)
    include_directories(~/envs/MNN/install/include)
    link_directories(~/envs/MNN/install/lib)

    add_library(yoloInferMNN SHARED 
        ${CMAKE_CURRENT_SOURCE_DIR}/platform/mnn/mnn.cpp
    )
    target_link_libraries(yoloInferMNN PUBLIC
        MNN
    )
endif()

if(ROCKCHIP)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

    add_library(yoloInferRKNN SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/platform/rockchip/rknn.cpp
    )

    target_link_libraries(yoloInferRKNN PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/librknnrt.so
    )

endif()

if (ASCEND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/platform/ascend)
endif()


if (HORIZON)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/platform/horizon)
endif()
