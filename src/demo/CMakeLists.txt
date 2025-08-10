if(WIN32)
    cmake_minimum_required(VERSION 4.1)
elseif(UNIX AND NOT APPLE)
    cmake_minimum_required(VERSION 3.10)
endif()

project(yoloInferDemo LANGUAGES CXX)

if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "-O3 -fPIC -w ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS -pthread)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (CROSS_COMPILE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
endif()

add_executable(detect
    ${CMAKE_CURRENT_SOURCE_DIR}/demo.cpp
)

target_link_libraries(detect
    YOLODetect
    yaml-cpp
    ${OpenCV_LIBRARIES} 
)


add_executable(detect_noshow
    ${CMAKE_CURRENT_SOURCE_DIR}/demo_no_show.cpp
)

target_link_libraries(detect_noshow
    YOLODetect
    yaml-cpp
    ${OpenCV_LIBRARIES} 
)

add_executable(detect_image
    ${CMAKE_CURRENT_SOURCE_DIR}/demo_image.cpp
)

target_link_libraries(detect_image
    YOLODetect
    yaml-cpp
    ${OpenCV_LIBRARIES} 
)

add_executable(detect_one_image
    ${CMAKE_CURRENT_SOURCE_DIR}/demo_one_image.cpp
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
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../models DESTINATION .)

if (ASCEND)

elseif (CROSS_COMPILE)
    install(DIRECTORY ${CROSS_COMPILE_LIBRARY_PATH} DESTINATION .)
endif()