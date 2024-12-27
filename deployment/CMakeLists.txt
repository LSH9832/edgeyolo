cmake_minimum_required(VERSION 3.0.2)

# set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl") 
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")


# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/bin/lib/")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/bin/")


option(TENSORRT "tensorrt platform" OFF)
option(MNN "MNN platform" OFF)

# 特定平台
option(ROCKCHIP "RockChip platform(RK3588)" OFF)
option(ASCEND "Ascend platform(Ascend310)" OFF)
option(HORIZON "Horizon platform(Journal5)" OFF)

option(ARM64 "use arm64, valid when cross-compile" OFF)
option(DEFAULT "force to use default compiler" OFF)
option(BUILD_API "build platform" OFF)

if(DEFAULT)
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++")
elseif(ARM64 OR ROCKCHIP OR HORIZON OR ASCEND)
    # sudo apt install gcc-aarch64-linux-gnu
    set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
endif()


project(yoloInferDemo LANGUAGES CXX)

add_definitions(-std=c++14)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS -pthread)
set(CMAKE_BUILD_TYPE Release)

add_subdirectory(yolo)

option(BUILD_DEMO "build demo" OFF)
option(CROSS_COMPILE "cross compile" OFF)

if(CROSS_COMPILE)
  set(CMAKE_SYSTEM_PROCESSOR "aarch64")
endif()

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/yolo)
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

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
    yolo/detect.cpp
  )


  target_link_libraries(YOLODetect PUBLIC
      ${OpenCV_LIBRARIES}
      yaml-cpp
      dl
  )


  install(TARGETS YOLODetect DESTINATION lib)
  install(DIRECTORY yolo/lib DESTINATION .)
  install(PROGRAMS yolo/detect.hpp DESTINATION include)


  install(DIRECTORY yolo DESTINATION python)
  install(PROGRAMS detect.py DESTINATION python)
endif()

if(BUILD_DEMO)
  # link_directories()
  if (CROSS_COMPILE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
  endif()


  add_executable(detect
      src/demo.cpp
  )

  target_link_libraries(detect
      YOLODetect
      yaml-cpp
      ${OpenCV_LIBRARIES} 
  )


  add_executable(detect_noshow
      src/demo_no_show.cpp
  )

  target_link_libraries(detect_noshow
      YOLODetect
      yaml-cpp
      ${OpenCV_LIBRARIES} 
  )

  add_executable(detect_image
      src/demo_image.cpp
  )

  target_link_libraries(detect_image
      YOLODetect
      yaml-cpp
      ${OpenCV_LIBRARIES} 
  )

  add_executable(detect_one_image
      src/demo_image.cpp
  )

  target_link_libraries(detect_one_image
      YOLODetect
      yaml-cpp
      ${OpenCV_LIBRARIES} 
  )

  install(TARGETS detect DESTINATION bin)
  install(TARGETS detect_noshow DESTINATION bin)
  install(TARGETS detect_image DESTINATION bin)
  install(TARGETS detect_one_image DESTINATION bin)
  install(DIRECTORY models DESTINATION .)
  if (ASCEND)
    
  elseif (CROSS_COMPILE)
    install(DIRECTORY ${CROSS_COMPILE_LIBRARY_PATH} DESTINATION .)
  endif()
endif()
