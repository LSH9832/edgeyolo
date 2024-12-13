cmake_minimum_required(VERSION 3.0.2)

project(yoloInferAscend)

if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
endif()

set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS -pthread)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")


set(CROSS_COMPILE_PATH "/usr/local/ubuntu_crossbuild_devkit/sysroot")
set(ASCEND_TOOLKIT_PATH "/usr/local/Ascend/ascend-toolkit/5.0.mdc300")

# set(CROSS_COMPILE_PATH "/home/lsh/code/CrossCompile/ascend/ubuntu_crossbuild_devkit/sysroot")
# set(ASCEND_TOOLKIT_PATH "/home/lsh/code/CrossCompile/ascend/Ascend/ascend-toolkit/5.0.mdc300")

set(ASCEND_ACLLIB_PATH ${ASCEND_TOOLKIT_PATH}/arm64-linux/acllib)
set(ASCEND_ATCLIB_PATH ${ASCEND_TOOLKIT_PATH}/x86_64-linux/atc)
set(ASCEND_DRIVER_PATH ${ASCEND_TOOLKIT_PATH}/x86_64-linux/mdc_driver_sdk)


include_directories(
    ${ASCEND_ACLLIB_PATH}/include
    ${ASCEND_DRIVER_PATH}/driver/include
    ${CROSS_COMPILE_PATH}/usr/local/include
    ${CROSS_COMPILE_PATH}/usr/include
    ${CROSS_COMPILE_PATH}/usr/include/aarch64-linux-gnu
)

link_directories(
    ${ASCEND_DRIVER_PATH}/driver/lib64
    ${ASCEND_ATCLIB_PATH}/lib64
    # ${ASCEND_ACLLIB_PATH}/lib64
    ${ASCEND_ACLLIB_PATH}/lib64/stub
    ${CROSS_COMPILE_PATH}/usr/local/lib
    ${CROSS_COMPILE_PATH}/lib
    ${CROSS_COMPILE_PATH}/lib/aarch64-linux-gnu
    ${CROSS_COMPILE_PATH}/usr/lib
    ${CROSS_COMPILE_PATH}/usr/lib/aarch64-linux-gnu
)


add_library(yoloInferAscend SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/davinci_net.cpp
)

target_link_libraries(yoloInferAscend PUBLIC
    # OMX_hisi_video_decoder OMX_hisi_video_encoder
    # Dvpp_png_decoder Dvpp_jpeg_decoder Dvpp_jpeg_encoder
    acl_dvpp
    ascendcl
    slog
    c_sec
    pthread
)