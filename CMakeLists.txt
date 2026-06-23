cmake_minimum_required(VERSION 3.24)

project(
    cuda_reinforcement_learning_ai
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(
    cuda_device_check
    cuda/src/main.cu
    cuda/src/cuda_check.cu
    cuda/src/gae.cu
)

target_include_directories(
    cuda_device_check
    PRIVATE
    cuda/include
)

set_target_properties(
    cuda_device_check
    PROPERTIES
    CUDA_ARCHITECTURES 89
)
