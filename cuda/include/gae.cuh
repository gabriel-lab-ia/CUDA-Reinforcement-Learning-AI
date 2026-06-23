#pragma once

#include <cstddef>

namespace cuda_rl {

struct GaeLaunchConfig {
    float gamma;
    float lambda;
    std::size_t length;
};

void compute_gae_single_trajectory(
    const float* rewards,
    const float* values,
    const float* dones,
    float* advantages,
    float* returns,
    GaeLaunchConfig config
);

}  // namespace cuda_rl
