#include "gae.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cuda_rl {
namespace {

__global__ void gae_single_trajectory_kernel(
    const float* rewards,
    const float* values,
    const float* dones,
    float* advantages,
    float* returns,
    float gamma,
    float lambda,
    std::size_t length
) {
    // Initial implementation targets correctness for one trajectory.
    // It is intentionally sequential inside one CUDA thread because GAE is a
    // reverse recurrence. Parallel segmented scans are a future optimization.
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    float running_advantage = 0.0F;
    float next_value = 0.0F;
    for (std::size_t offset = 0; offset < length; ++offset) {
        const std::size_t index = length - 1 - offset;
        const float non_terminal = 1.0F - dones[index];
        const float delta =
            rewards[index] + gamma * next_value * non_terminal - values[index];
        running_advantage =
            delta + gamma * lambda * non_terminal * running_advantage;
        advantages[index] = running_advantage;
        returns[index] = running_advantage + values[index];
        next_value = values[index];
    }
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

}  // namespace

void compute_gae_single_trajectory(
    const float* rewards,
    const float* values,
    const float* dones,
    float* advantages,
    float* returns,
    GaeLaunchConfig config
) {
    if (config.length == 0) {
        throw std::invalid_argument("GAE length must be positive.");
    }
    gae_single_trajectory_kernel<<<1, 1>>>(
        rewards,
        values,
        dones,
        advantages,
        returns,
        config.gamma,
        config.lambda,
        config.length
    );
    check_cuda(cudaGetLastError(), "launch gae_single_trajectory_kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize gae_single_trajectory_kernel");
}

}  // namespace cuda_rl
