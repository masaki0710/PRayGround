#include "field_smooth.cuh"
#include <cstdio>

namespace prayground {

static inline void cudaCheck(cudaError_t err, const char* call, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("CUDA call (%s) failed with error: '%s' (%s:%d)\n", call, cudaGetErrorString(err), file, line);
    }
}

#define CUDA_CHECK_LOCAL(call) cudaCheck((call), #call, __FILE__, __LINE__)

__global__ void smoothFieldStep(const float* in_field, float* out_field, int nx, int ny, int nz, float weight)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nxy = nx * ny;
    const int total = nxy * nz;
    if (idx >= total)
        return;

    const int x = idx % nx;
    const int y = (idx / nx) % ny;
    const int z = idx / nxy;

    if (x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1) {
        out_field[idx] = in_field[idx];
        return;
    }

    const float center = in_field[idx];
    const float nsum =
        in_field[idx - 1] + in_field[idx + 1] +
        in_field[idx - nx] + in_field[idx + nx] +
        in_field[idx - nxy] + in_field[idx + nxy];
    const float avg = nsum / 6.0f;
    out_field[idx] = center + weight * (avg - center);
}

extern "C" void smoothScalarFieldCUDAInPlace(
    float* d_field,
    int nx,
    int ny,
    int nz,
    int iters,
    float weight
)
{
    if (!d_field || iters <= 0 || nx <= 0 || ny <= 0 || nz <= 0)
        return;

    const int nxy = nx * ny;
    const int total = nxy * nz;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    float* d_temp = nullptr;
    CUDA_CHECK_LOCAL(cudaMalloc(&d_temp, sizeof(float) * total));

    float* in_field = d_field;
    float* out_field = d_temp;
    for (int i = 0; i < iters; ++i) {
        smoothFieldStep<<<blocks, threads>>>(in_field, out_field, nx, ny, nz, weight);
        float* tmp = in_field;
        in_field = out_field;
        out_field = tmp;
    }

    if (in_field != d_field) {
        CUDA_CHECK_LOCAL(cudaMemcpy(d_field, in_field, sizeof(float) * total, cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK_LOCAL(cudaFree(d_temp));
}

} // namespace prayground
