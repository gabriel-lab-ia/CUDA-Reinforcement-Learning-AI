#include "cuda_check.cuh"

#include <cstdlib>
#include <iostream>

namespace {

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::cerr
            << "CUDA error during "
            << operation
            << ": "
            << cudaGetErrorString(result)
            << '\n';

        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

void print_cuda_device_info() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

    std::cout << "CUDA devices detected: " << device_count << '\n';

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp properties{};

        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties"
        );

        std::cout
            << "Device " << device << ": " << properties.name << '\n'
            << "Compute capability: "
            << properties.major << '.' << properties.minor << '\n'
            << "Global memory: "
            << properties.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)
            << " GiB\n"
            << "Multiprocessors: "
            << properties.multiProcessorCount
            << '\n';
    }
}
