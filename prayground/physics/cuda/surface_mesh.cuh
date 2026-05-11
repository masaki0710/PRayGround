#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <prayground/core/aabb.h>
#include <prayground/math/vec.h>

namespace prayground {
    void extractSurfaceTrianglesCUDA(
        const float* d_field,
        int nx,
        int ny,
        int nz,
        const AABB& wall,
        float iso_level,
        std::vector<Vec3f>& out_triangle_vertices,
        int max_triangles = 0,
        cudaStream_t stream = 0
    );
} // namespace prayground
