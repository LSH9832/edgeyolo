if(WIN32)
  cmake_minimum_required(VERSION 4.1)
elseif(UNIX AND NOT APPLE)
  cmake_minimum_required(VERSION 3.10)
endif()


option(TENSORRT "tensorrt platform" OFF)
option(MNN "MNN platform" OFF)
option(OPENVINO "OpenVINO platform" OFF)

# 特定平台
option(RKNN "RKNN for RockChip platform(RK3588/3576/3568)" OFF)
option(ASCEND "Ascend platform(Ascend310)" OFF)
option(HBDNN "hbDNN for Horizon BPU platform(Journal5)" OFF)

option(ARM64 "use arm64, valid when cross-compile" OFF)
option(DEFAULT "force to use default compiler" OFF)
option(BUILD_API "build platform" OFF)

if (NOT WIN32)
  if(DEFAULT)
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++")
  elseif(ARM64 OR ROCKCHIP OR HORIZON OR ASCEND)
    # sudo apt install gcc-aarch64-linux-gnu
    set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
  endif()
endif()


project(yoloInferDemo LANGUAGES CXX)

if(OPENVINO)
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 14)
endif()

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "-O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS -pthread)
  set(CMAKE_BUILD_TYPE Release)
endif()
include_directories(
  ./include
  ./src
)
add_subdirectory(src/yolo)

option(BUILD_DEMO "build demo" OFF)
option(CROSS_COMPILE "cross compile" OFF)

if(CROSS_COMPILE)
  set(CMAKE_SYSTEM_PROCESSOR "aarch64")
endif()

if(WIN32)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${CMAKE_SYSTEM_NAME}/yolo)
else()
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/yolo)
endif()


set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
if(WIN32)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/bin")
else()
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()
# message("CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")


if(BUILD_DEMO OR BUILD_API)
  
  # opencv
  if (ASCEND)
    include_directories(/usr/local/ubuntu_crossbuild_devkit/sysroot/usr/include)
    link_directories(/usr/local/ubuntu_crossbuild_devkit/sysroot/usr/local/lib)
    set(OpenCV_LIBRARIES "opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_stitching;opencv_video;opencv_videoio;opencv_shape;opencv_superres;opencv_videostab;opencv_viz;")

  elseif(CROSS_COMPILE)
    set(CROSS_COMPILE_PACKAGE_PATH /home/lsh/code/CrossCompile/install/aarch64)
    set(CROSS_COMPILE_INCLUDE_PATH ${CROSS_COMPILE_PACKAGE_PATH}/include)
    set(CROSS_COMPILE_LIBRARY_PATH ${CROSS_COMPILE_PACKAGE_PATH}/lib)

    find_package(OpenCV HINTS ${CROSS_COMPILE_PACKAGE_PATH})
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${CROSS_COMPILE_LIBRARY_PATH})
  else()
    include_directories(/usr/include/opencv4)
    find_package(OpenCV 4 REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
  endif()


  # message("${OpenCV_LIBRARIES}")


  add_library(YOLODetect SHARED
    src/yolo/detect.cpp
  )


  target_link_libraries(YOLODetect PUBLIC
      ${OpenCV_LIBRARIES}
      yaml-cpp
      dl
  )

  if(WIN32)
    install(TARGETS YOLODetect DESTINATION bin)
  endif()
  install(TARGETS YOLODetect DESTINATION lib)
  install(DIRECTORY python/yolo/lib DESTINATION .)
  install(PROGRAMS include/yolo/detect.hpp DESTINATION include/yolo)

  install(DIRECTORY python/yolo DESTINATION python)
  install(PROGRAMS python/detect.py DESTINATION python)
endif()

if(BUILD_DEMO)
  add_subdirectory(src/demo)
endif()
