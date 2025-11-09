cmake_minimum_required(VERSION 3.10)
project(MultispectralProcessor)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)

add_executable(MultispectralProcessor 
    src/main.cpp 
    src/ImageProcessor.cpp
    src/ROISelector.cpp
)

target_link_libraries(MultispectralProcessor ${OpenCV_LIBS})