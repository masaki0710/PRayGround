#include "surface_mesh.cuh"
#include <prayground/optix/macros.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace marching_cubes_tables {
    extern const int kTriTable[256][16];
}

namespace prayground {
namespace {

DEVICE inline int gridIndex(int x, int y, int z, int nx, int ny)
{
    return x + nx * (y + ny * z);
}

DEVICE inline Vec3f lerpIso(const Vec3f& p0, const Vec3f& p1, float v0, float v1, float iso)
{
    const float denom = (v1 - v0);
    float t = (fabsf(denom) < 1e-7f) ? 0.5f : (iso - v0) / denom;
    t = fminf(1.0f, fmaxf(0.0f, t));
    return p0 + (p1 - p0) * t;
}

__device__ inline void emitTriangle(
    Vec3f* out_vertices,
    int tri_idx,
    const Vec3f& a,
    const Vec3f& b,
    const Vec3f& c)
{
    const int base = tri_idx * 3;
    out_vertices[base + 0] = a;
    out_vertices[base + 1] = b;
    out_vertices[base + 2] = c;
}

__device__ inline void emitTriangleIfRoom(
    Vec3f* out_vertices,
    int* tri_count,
    int max_triangles,
    const Vec3f& a,
    const Vec3f& b,
    const Vec3f& c)
{
    const int tri_idx = atomicAdd(tri_count, 1);
    if (tri_idx >= max_triangles)
        return;
    emitTriangle(out_vertices, tri_idx, a, b, c);
}

__device__ inline Vec3f interpolateCubeEdge(
    const Vec3f cp[8],
    const float cv[8],
    int edge,
    float iso)
{
    static constexpr int edge_to_vertex[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };
    const int v0 = edge_to_vertex[edge][0];
    const int v1 = edge_to_vertex[edge][1];
    return lerpIso(cp[v0], cp[v1], cv[v0], cv[v1], iso);
}

__device__ __constant__ int d_tri_table[256][16];

__device__ inline void polygonizeCube(
    const Vec3f cp[8],
    const float cv[8],
    float iso,
    Vec3f* out_vertices,
    int* tri_count,
    int max_triangles)
{
    int case_index = 0;
    if (cv[0] >= iso) case_index |= 1;
    if (cv[1] >= iso) case_index |= 2;
    if (cv[2] >= iso) case_index |= 4;
    if (cv[3] >= iso) case_index |= 8;
    if (cv[4] >= iso) case_index |= 16;
    if (cv[5] >= iso) case_index |= 32;
    if (cv[6] >= iso) case_index |= 64;
    if (cv[7] >= iso) case_index |= 128;

    const int* tri_edges = d_tri_table[case_index];
    for (int i = 0; tri_edges[i] != -1; i += 3) {
        const Vec3f a = interpolateCubeEdge(cp, cv, tri_edges[i + 0], iso);
        const Vec3f b = interpolateCubeEdge(cp, cv, tri_edges[i + 1], iso);
        const Vec3f c = interpolateCubeEdge(cp, cv, tri_edges[i + 2], iso);
        emitTriangleIfRoom(out_vertices, tri_count, max_triangles, a, b, c);
    }
}

__global__ void polygonizeSurfaceFieldKernel(
    const float* field,
    int nx,
    int ny,
    int nz,
    Vec3f minp,
    Vec3f step,
    float iso,
    Vec3f* out_vertices,
    int max_triangles,
    int* tri_count)
{
    const int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = (nx - 1) * (ny - 1) * (nz - 1);
    if (cell_idx >= cell_count)
        return;

    const int nxy = (nx - 1) * (ny - 1);
    const int z = cell_idx / nxy;
    const int rem = cell_idx - z * nxy;
    const int y = rem / (nx - 1);
    const int x = rem - y * (nx - 1);

    Vec3f cp[8];
    float cv[8];
    cp[0] = Vec3f(minp.x() + step.x() * x,     minp.y() + step.y() * y,     minp.z() + step.z() * z);
    cp[1] = Vec3f(minp.x() + step.x() * (x+1), minp.y() + step.y() * y,     minp.z() + step.z() * z);
    cp[2] = Vec3f(minp.x() + step.x() * (x+1), minp.y() + step.y() * (y+1), minp.z() + step.z() * z);
    cp[3] = Vec3f(minp.x() + step.x() * x,     minp.y() + step.y() * (y+1), minp.z() + step.z() * z);
    cp[4] = Vec3f(minp.x() + step.x() * x,     minp.y() + step.y() * y,     minp.z() + step.z() * (z+1));
    cp[5] = Vec3f(minp.x() + step.x() * (x+1), minp.y() + step.y() * y,     minp.z() + step.z() * (z+1));
    cp[6] = Vec3f(minp.x() + step.x() * (x+1), minp.y() + step.y() * (y+1), minp.z() + step.z() * (z+1));
    cp[7] = Vec3f(minp.x() + step.x() * x,     minp.y() + step.y() * (y+1), minp.z() + step.z() * (z+1));

    cv[0] = field[gridIndex(x, y, z, nx, ny)];
    cv[1] = field[gridIndex(x + 1, y, z, nx, ny)];
    cv[2] = field[gridIndex(x + 1, y + 1, z, nx, ny)];
    cv[3] = field[gridIndex(x, y + 1, z, nx, ny)];
    cv[4] = field[gridIndex(x, y, z + 1, nx, ny)];
    cv[5] = field[gridIndex(x + 1, y, z + 1, nx, ny)];
    cv[6] = field[gridIndex(x + 1, y + 1, z + 1, nx, ny)];
    cv[7] = field[gridIndex(x, y + 1, z + 1, nx, ny)];

    polygonizeCube(cp, cv, iso, out_vertices, tri_count, max_triangles);
}

static inline void cudaCheck(cudaError_t err, const char* call, const char* file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA call (%s) failed with error: '%s' (%s:%d)\n", call, cudaGetErrorString(err), file, line);
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

#define CUDA_CHECK_LOCAL(call) cudaCheck((call), #call, __FILE__, __LINE__)

static void ensureMarchingCubesTablesUploaded(cudaStream_t stream)
{
    static bool uploaded = false;
    if (uploaded)
        return;

    CUDA_CHECK_LOCAL(cudaMemcpyToSymbolAsync(
        d_tri_table,
        ::marching_cubes_tables::kTriTable,
        sizeof(::marching_cubes_tables::kTriTable),
        0,
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK_LOCAL(cudaStreamSynchronize(stream));
    uploaded = true;
}

} // namespace

void extractSurfaceTrianglesCUDA(
    const float* d_field,
    int nx,
    int ny,
    int nz,
    const AABB& wall,
    float iso_level,
    std::vector<Vec3f>& out_triangle_vertices,
    int max_triangles,
    cudaStream_t stream
)
{
    out_triangle_vertices.clear();
    if (!d_field || nx <= 1 || ny <= 1 || nz <= 1)
        return;

    const Vec3f minp = wall.min();
    const Vec3f maxp = wall.max();
    const Vec3f extent = maxp - minp;
    const Vec3f step(
        extent.x() / (float)(nx - 1),
        extent.y() / (float)(ny - 1),
        extent.z() / (float)(nz - 1)
    );

    ensureMarchingCubesTablesUploaded(stream);

    const int cell_count = (nx - 1) * (ny - 1) * (nz - 1);
    const int capacity = (max_triangles > 0) ? max_triangles : cell_count * 5;
    out_triangle_vertices.resize((size_t)capacity * 3);

    Vec3f* d_vertices = nullptr;
    int* d_tri_count = nullptr;
    CUDA_CHECK_LOCAL(cudaMalloc(&d_vertices, sizeof(Vec3f) * out_triangle_vertices.size()));
    CUDA_CHECK_LOCAL(cudaMalloc(&d_tri_count, sizeof(int)));
    CUDA_CHECK_LOCAL(cudaMemsetAsync(d_tri_count, 0, sizeof(int), stream));

    const int threads = 256;
    const int blocks = (cell_count + threads - 1) / threads;
    polygonizeSurfaceFieldKernel<<<blocks, threads, 0, stream>>>(
        d_field,
        nx,
        ny,
        nz,
        minp,
        step,
        iso_level,
        d_vertices,
        capacity,
        d_tri_count
    );
    CUDA_CHECK_LOCAL(cudaGetLastError());

    int tri_count = 0;
    CUDA_CHECK_LOCAL(cudaMemcpyAsync(&tri_count, d_tri_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_LOCAL(cudaStreamSynchronize(stream));

    const size_t tri_vertices = (size_t)std::max(0, tri_count) * 3;
    out_triangle_vertices.resize(tri_vertices);
    if (tri_vertices > 0) {
        CUDA_CHECK_LOCAL(cudaMemcpy(out_triangle_vertices.data(), d_vertices, sizeof(Vec3f) * tri_vertices, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK_LOCAL(cudaFree(d_vertices));
    CUDA_CHECK_LOCAL(cudaFree(d_tri_count));
}

} // namespace prayground
