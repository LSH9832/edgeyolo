if(WIN32)
    cmake_minimum_required(VERSION 4.1)
elseif(UNIX AND NOT APPLE)
    cmake_minimum_required(VERSION 3.10)
endif()

project(yoloinferMNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "-O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS -pthread)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../python/yolo/lib)

add_library(yoloInferRKNN SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/rknn.cpp
)

target_link_libraries(yoloInferRKNN PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../python/yolo/lib/librknnrt.so
)