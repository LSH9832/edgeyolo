cmake_minimum_required(VERSION 3.14)

project(yolo LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(PROJECT_SOURCES
    main.cpp
    trtengine.cpp
    argparse.h
    trtengine.h
    logging.h
)

file(GLOB_RECURSE _HEAD ${CMAKE_CURRENT_LIST_DIR}/*.h )
file(GLOB_RECURSE _CPP ${CMAKE_CURRENT_LIST_DIR}/*.cpp )

include_directories(D:/software/GPU/CUDA/v11.1/development/include)
link_directories(D:/software/GPU/CUDA/v11.1/development/lib)
link_directories(D:/software/GPU/CUDA/v11.1/development/lib/x64)


include_directories(D:/software/Programming/opencv/build/include)
link_directories(D:/software/Programming/opencv/build/x64/vc15/lib)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolo
    ${_HEAD}
    ${_CPP}
    main.cpp
)

target_link_libraries(yolo nvinfer)
target_link_libraries(yolo cudart)
target_link_libraries(yolo ${OpenCV_LIBS})
target_link_libraries(yolo Qt${QT_VERSION_MAJOR}::Core)

install(TARGETS yolo
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
