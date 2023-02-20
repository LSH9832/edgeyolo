cmake_minimum_required(VERSION 2.6)

project(yolo LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(PROJECT_SOURCES
    main.cpp
    trtengine.cpp
    argparse.h
    trtengine.h
    logging.h
)

file(GLOB _HEAD ${CMAKE_CURRENT_LIST_DIR}/*.h )
file(GLOB _CPP ${CMAKE_CURRENT_LIST_DIR}/*.cpp )

# cuda # cudnn # tensorrt
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolo
    ${_HEAD}
    ${_CPP}
)

target_link_libraries(yolo nvinfer)
target_link_libraries(yolo cudart)
target_link_libraries(yolo ${OpenCV_LIBS})
target_link_libraries(yolo Qt${QT_VERSION_MAJOR}::Core)
target_link_libraries(yolo Qt${QT_VERSION_MAJOR}::Widgets)
