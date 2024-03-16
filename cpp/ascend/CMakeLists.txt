cmake_minimum_required(VERSION 2.8)

set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")

project(ascend)


set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE "Release")


set(ASCEND_ACLLIB_PATH "/usr/local/Ascend/ascend-toolkit/5.0.mdc300/acllib")
set(ASCEND_DRIVER_PATH "/usr/local/Ascend/ascend-toolkit/5.0.mdc300/x86_64-linux/mdc_driver_sdk")
set(THIRD_PARTY_PATH "/root/MDC/thrid_party")
set(CROSS_COMPILE_PATH "/usr/local/ubuntu_crossbuild_devkit/sysroot/usr")


include_directories(
    ${ASCEND_ACLLIB_PATH}/include
    ${ASCEND_DRIVER_PATH}/driver/include
    ${CROSS_COMPILE_PATH}/local/include
    ${CROSS_COMPILE_PATH}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
)

link_directories(
    ${ASCEND_DRIVER_PATH}/driver/lib64
    ${ASCEND_ACLLIB_PATH}/lib64/stub
    ${CROSS_COMPILE_PATH}/local/lib
    /usr/local/ubuntu_crossbuild_devkit/sysroot/lib
    /usr/local/ubuntu_crossbuild_devkit/sysroot/lib/aarch64-linux-gnu
    ${CROSS_COMPILE_PATH}/lib
    ${CROSS_COMPILE_PATH}/lib/aarch64-linux-gnu
)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


add_executable(${PROJECT_NAME}_det
    src/main.cpp 
    src/dvpp_handler.cpp
    src/davinci_net.cpp
)

target_link_libraries(${PROJECT_NAME}_det
    yaml-cpp
    opencv_calib3d opencv_core opencv_dnn opencv_features2d opencv_flann opencv_highgui opencv_imgcodecs opencv_imgproc opencv_ml opencv_objdetect opencv_photo opencv_stitching opencv_video opencv_videoio opencv_shape opencv_superres opencv_videostab opencv_viz 
    Dvpp_api Dvpp_vpc
    OMX_hisi_video_decoder OMX_hisi_video_encoder
    Dvpp_png_decoder Dvpp_jpeg_decoder Dvpp_jpeg_encoder
    acl_dvpp ascendcl
    slog
    c_sec
    pthread
)

