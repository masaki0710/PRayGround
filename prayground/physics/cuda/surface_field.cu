#include "surface_field.cuh"
#include "field_smooth.cuh"
#include <prayground/optix/macros.h>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace prayground {
namespace {

DEVICE inline int gridIndex(int x, int y, int z, int nx, int ny)
{
    return x + nx * (y + ny * z);
}

DEVICE inline float kernelWeight(float distance, float support)
{
    if (distance >= support)
        return 0.0f;
    const float q = 1.0f - (distance / support);
    return q * q;
}

__global__ void buildSurfaceFieldKernel(
    const SPHParticles::Data* particles,
    uint32_t num_particles,
    Vec3f minp,
    Vec3f step,
    int nx,
    int ny,
    int nz,
    float support,
    float* field)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (idx >= total)
        return;

    const int nxy = nx * ny;
    const int z = idx / nxy;
    const int rem = idx - z * nxy;
    const int y = rem / nx;
    const int x = rem - y * nx;

    const Vec3f p(
        minp.x() + step.x() * (float)x,
        minp.y() + step.y() * (float)y,
        minp.z() + step.z() * (float)z
    );

    float phi = 0.0f;
    for (uint32_t i = 0; i < num_particles; ++i) {
        const float distance = length(p - particles[i].position);
        phi += kernelWeight(distance, support);
    }

    field[idx] = phi;
}

static inline void cudaCheck(cudaError_t err, const char* call, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("CUDA call (%s) failed with error: '%s' (%s:%d)\n", call, cudaGetErrorString(err), file, line);
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

#define CUDA_CHECK_LOCAL(call) cudaCheck((call), #call, __FILE__, __LINE__)

} // namespace

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
    float** out_device_field,
    cudaStream_t stream
)
{
    out_field.assign((size_t)nx * ny * nz, 0.0f);
    if (!d_particles || num_particles == 0 || nx <= 0 || ny <= 0 || nz <= 0)
        return;

    SPHParticles::Data first_particle{};
    CUDA_CHECK_LOCAL(cudaMemcpy(&first_particle, d_particles, sizeof(SPHParticles::Data), cudaMemcpyDeviceToHost));
    const float radius = first_particle.radius;
    const float support = std::max(1e-3f, support_radius_scale * radius);
    const Vec3f minp = wall.min();
    const Vec3f maxp = wall.max();
    const Vec3f extent = maxp - minp;

    const Vec3f step(
        extent.x() / (float)(nx - 1),
        extent.y() / (float)(ny - 1),
        extent.z() / (float)(nz - 1)
    );

    const size_t bytes = sizeof(float) * out_field.size();
    float* d_field = nullptr;
    CUDA_CHECK_LOCAL(cudaMalloc(&d_field, bytes));

    const int total = nx * ny * nz;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    buildSurfaceFieldKernel<<<blocks, threads, 0, stream>>>(
        d_particles,
        num_particles,
        minp,
        step,
        nx,
        ny,
        nz,
        support,
        d_field
    );
    CUDA_CHECK_LOCAL(cudaGetLastError());

    if (field_smooth_iters > 0) {
        smoothScalarFieldCUDAInPlace(d_field, nx, ny, nz, field_smooth_iters, field_smooth_weight);
    }

    CUDA_CHECK_LOCAL(cudaMemcpyAsync(out_field.data(), d_field, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_LOCAL(cudaStreamSynchronize(stream));
    if (out_device_field) {
        *out_device_field = d_field;
    } else {
        CUDA_CHECK_LOCAL(cudaFree(d_field));
    }
}

} // namespace prayground
