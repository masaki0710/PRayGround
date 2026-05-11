#pragma once

#include <cuda_runtime.h>

namespace prayground {
    // GPU Laplacian smoothing for a scalar field in a dense grid.
    extern "C" void smoothScalarFieldCUDAInPlace(
        float* d_field,
        int nx,
        int ny,
        int nz,
        int iters,
        float weight
    );
} // namespace prayground
