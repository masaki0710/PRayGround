#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <prayground/shape/trianglemesh.h>

namespace prayground {
    void computeTriangleMeshSmoothNormalsCUDA(
        const std::vector<Vec3f>& vertices,
        const std::vector<Face>& faces,
        std::vector<Vec3f>& out_normals,
        cudaStream_t stream = 0
    );
} // namespace prayground
