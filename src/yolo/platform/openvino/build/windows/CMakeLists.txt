cmake_minimum_required(VERSION 4.1)
project(yoloinferOpenVINO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)


include_directories(
    "D:/Software/coding/env/openvino_toolkit_windows/runtime/include"
)

link_directories(
    "D:/Software/coding/env/openvino_toolkit_windows/runtime/bin/intel64/Release"
    "D:/Software/coding/env/openvino_toolkit_windows/runtime/lib/intel64/Release"
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

install(TARGETS ${PROJECT_NAME} DESTINATION bin)