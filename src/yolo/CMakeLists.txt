if(WIN32)
    cmake_minimum_required(VERSION 4.1)
elseif(UNIX AND NOT APPLE)
    cmake_minimum_required(VERSION 3.10)
endif()

project(yoloInfer LANGUAGES CXX)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)

if (WIN32)
else()
    set(CMAKE_CXX_FLAGS "-O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS -pthread)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(OPENVINO)
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 14)
endif()

if(WIN32)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../../python/yolo/lib/")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../../python/yolo/lib/")
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/../../python/yolo/lib/")
endif()


if (OPENVINO)
    if(WIN32)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/platform/openvino/build/windows)
    else()
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/platform/openvino/build/linux)
    endif()
endif()

if (MNN)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/platform/mnn)
endif()

if(ASCEND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/platform/ascend)
endif()

if(RKNN)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/platform/rknn)
endif()

if(HBDNN)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/platform/hbdnn)
endif()