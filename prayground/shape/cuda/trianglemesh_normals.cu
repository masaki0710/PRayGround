#include "trianglemesh_normals.cuh"
#include <prayground/optix/macros.h>
#include <cstdio>
#include <stdexcept>

namespace prayground {
namespace {

DEVICE inline void atomicAddVec3(Vec3f& dst, const Vec3f& v)
{
    atomicAdd(&dst.x(), v.x());
    atomicAdd(&dst.y(), v.y());
    atomicAdd(&dst.z(), v.z());
}

__global__ void accumulateSmoothNormalsKernel(
    const Vec3f* vertices,
    const Face* faces,
    uint32_t num_faces,
    Vec3f* normals)
{
    const uint32_t face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_faces)
        return;

    const Face face = faces[face_idx];
    const Vec3f p0 = vertices[face.vertex_id.x()];
    const Vec3f p1 = vertices[face.vertex_id.y()];
    const Vec3f p2 = vertices[face.vertex_id.z()];
    const Vec3f face_normal = cross(p1 - p0, p2 - p0);

    atomicAddVec3(normals[face.vertex_id.x()], face_normal);
    atomicAddVec3(normals[face.vertex_id.y()], face_normal);
    atomicAddVec3(normals[face.vertex_id.z()], face_normal);
}

__global__ void normalizeNormalsKernel(Vec3f* normals, uint32_t num_vertices)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices)
        return;

    Vec3f n = normals[idx];
    const float len = length(n);
    normals[idx] = (len > 1e-8f) ? (n / len) : Vec3f(0.0f);
}

static inline void cudaCheck(cudaError_t err, const char* call, const char* file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA call (%s) failed with error: '%s' (%s:%d)\n", call, cudaGetErrorString(err), file, line);
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

#define CUDA_CHECK_LOCAL(call) cudaCheck((call), #call, __FILE__, __LINE__)

} // namespace

void computeTriangleMeshSmoothNormalsCUDA(
    const std::vector<Vec3f>& vertices,
    const std::vector<Face>& faces,
    std::vector<Vec3f>& out_normals,
    cudaStream_t stream
)
{
    out_normals.assign(vertices.size(), Vec3f(0.0f));
    if (vertices.empty() || faces.empty())
        return;

    const size_t vertices_bytes = sizeof(Vec3f) * vertices.size();
    const size_t faces_bytes = sizeof(Face) * faces.size();

    Vec3f* d_vertices = nullptr;
    Face* d_faces = nullptr;
    Vec3f* d_normals = nullptr;

    CUDA_CHECK_LOCAL(cudaMalloc(&d_vertices, vertices_bytes));
    CUDA_CHECK_LOCAL(cudaMalloc(&d_faces, faces_bytes));
    CUDA_CHECK_LOCAL(cudaMalloc(&d_normals, vertices_bytes));

    CUDA_CHECK_LOCAL(cudaMemcpyAsync(d_vertices, vertices.data(), vertices_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_LOCAL(cudaMemcpyAsync(d_faces, faces.data(), faces_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_LOCAL(cudaMemsetAsync(d_normals, 0, vertices_bytes, stream));

    const int threads = 256;
    const int face_blocks = ((int)faces.size() + threads - 1) / threads;
    accumulateSmoothNormalsKernel<<<face_blocks, threads, 0, stream>>>(d_vertices, d_faces, (uint32_t)faces.size(), d_normals);
    CUDA_CHECK_LOCAL(cudaGetLastError());

    const int vertex_blocks = ((int)vertices.size() + threads - 1) / threads;
    normalizeNormalsKernel<<<vertex_blocks, threads, 0, stream>>>(d_normals, (uint32_t)vertices.size());
    CUDA_CHECK_LOCAL(cudaGetLastError());

    CUDA_CHECK_LOCAL(cudaMemcpyAsync(out_normals.data(), d_normals, vertices_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_LOCAL(cudaStreamSynchronize(stream));

    CUDA_CHECK_LOCAL(cudaFree(d_vertices));
    CUDA_CHECK_LOCAL(cudaFree(d_faces));
    CUDA_CHECK_LOCAL(cudaFree(d_normals));
}

} // namespace prayground
