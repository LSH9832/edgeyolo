cmake_minimum_required(VERSION 3.10)
project(yoloinferOpenVINO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS -pthread)
set(CMAKE_BUILD_TYPE Release)


include_directories(
    "/mnt/e/packages/openvino_toolkit/runtime/include"
)

link_directories(
    "/mnt/e/packages/openvino_toolkit/runtime/lib/intel64"
)

add_library(${PROJECT_NAME} SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../openvino.cpp
)

target_link_libraries(${PROJECT_NAME}
    openvino
    openvino_c
    openvino_onnx_frontend
    openvino_paddle_frontend
    openvino_pytorch_frontend
    openvino_tensorflow_frontend
    openvino_tensorflow_lite_frontend
)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)