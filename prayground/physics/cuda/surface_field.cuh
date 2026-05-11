#pragma once

#include <cstdint>
#include <vector>
#include <prayground/physics/sph.h>
#include <prayground/core/aabb.h>
#include <cuda_runtime.h>

namespace prayground {
    void buildSurfaceFieldCUDA(
        const SPHParticles::Data* d_particles,
        uint32_t num_particles,
        const AABB& wall,
        int nx,
        int ny,
        int nz,
        float support_radius_scale,
        int field_smooth_iters,
        float field_smooth_weight,
        std::vector<float>& out_field,
        float** out_device_field = nullptr,
        cudaStream_t stream = 0
    );
} // namespace prayground
