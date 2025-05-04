cmake_minimum_required(VERSION 3.25)

project(bubble LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set (SOURCES "csrc/reduce.cu")
set(CMAKE_CUDA_ARCHITECTURES native)
add_library(bubble SHARED ${SOURCES})

set_target_properties(bubble PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)