cmake_minimum_required(VERSION 3.0.2)

project(yoloInferHorizon)

if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
endif()

set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS -pthread)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/horizon
)


add_library(yoloInferHorizon SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/horizon.cpp
)


target_link_libraries(yoloInferHorizon PUBLIC
    dnn
    pthread
)