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

include_directories(~/envs/MNN/install/include)
link_directories(~/envs/MNN/install/lib)

add_library(yoloInferMNN SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/mnn.cpp
)

target_link_libraries(yoloInferMNN PUBLIC
    MNN
)