cmake_minimum_required(VERSION 3.0.2)
project(mnn_det)


include_directories(
  include
)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(/usr/include/eigen3)

# --------------modify your own mnn path---------------
include_directories(/home/lsh/code/mnn_dev/MNN/include)
link_directories(/home/lsh/code/mnn_dev/MNN/build)
# --------------modify your own mnn path---------------


add_executable(mnn_det
  src/demo.cpp
)

target_link_libraries(mnn_det ${OpenCV_LIBRARIES})
target_link_libraries(mnn_det MNN)
target_link_libraries(mnn_det yaml-cpp)