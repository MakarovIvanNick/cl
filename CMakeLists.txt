cmake_minimum_required(VERSION 3.26)
project(untitled)

#-DCMAKE_PREFIX_PATH=E:\\opencv\\build E:\\cv_src\\sources\\cmake-build-debug

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "E:\\OpenCV-MinGW-Build-OpenCV-4.5.2-x64")

#add_executable(untitled main.cpp)
add_executable(untitled test.cpp)

find_package(OpenCL REQUIRED)
target_link_libraries(untitled OpenCL::OpenCL)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(untitled ${OpenCV_LIBS})