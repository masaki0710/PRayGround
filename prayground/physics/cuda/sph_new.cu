// Minimal, self-contained SPH implementation for real-time use
// - Density, Pressure, Pressure force (symmetric), Laplacian viscosity
// - Pairwise symmetric collision (i<j) with atomic updates
// - Wall handling with penalty + hard clamp + restitution

#include <prayground/physics/sph.h>
#include <prayground/math/util.h>
#include <prayground/math/vec.h>
#include <prayground/optix/macros.h>

namespace prayground {

    // Small helpers
    DEVICE inline float sqr(float x) { return x * x; }
    DEVICE inline float cube(float x) { return x * x * x; }

    // Reuse cubic spline kernels from existing code style
    DEVICE float cubicSpline(float q)
    {
        if ((0.0f <= q) && (q <= 0.5f))
            return 6.0f * (cube(q) - sqr(q)) + 1.0f;
        else if ((0.5f < q) && (q <= 1.0f))
            return 2.0f * cube(1.0f - q);
        else
            return 0.0f;
    }

    DEVICE float cubicSplineDerivative(float q)
    {
        if ((0.0f <= q) && (q <= 0.5f))
            return 6.0f * (3.0f * sqr(q) - 2.0f * q);
        else if ((0.5f < q) && (q <= 1.0f))
            return -6.0f * sqr(1.0f - q);
        else
            return 0.0f;
    }

    DEVICE float kernelW(float r, float h)
    {
        float q = r / h;
        float norm = 8.0f / (math::pi * cube(h));
        return norm * cubicSpline(q);
    }

    DEVICE float kernelGradScalar(float r, float h)
    {
        float q = r / h;
        float norm = 8.0f / (math::pi * cube(h));
        return norm * cubicSplineDerivative(q) / h; // dW/dr
    }

    DEVICE float viscosityLaplacian(float r, float h)
    {
        if (r < 0.0f || r > h) return 0.0f;
        const float h3 = cube(h);
        return 45.0f / (math::pi * h3 * h3) * (h - r);
    }

    // Atomic add for Vec3f components
    DEVICE inline void atomicAddVec3(Vec3f& dst, const Vec3f& v)
    {
        atomicAdd(&dst.x(), v.x());
        atomicAdd(&dst.y(), v.y());
        atomicAdd(&dst.z(), v.z());
    }

    // --- Kernels (naive O(n^2), simple and robust) ---

    extern "C" GLOBAL void computeDensityNew(SPHParticles::Data* particles, uint32_t n, SPHConfig config)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ((uint32_t)i >= n) return;

        SPHParticles::Data& pi = particles[i];
        const float h = config.kernel_size;

        // Reset density and include self contribution
        pi.density = pi.mass * kernelW(0.0f, h);

        for (uint32_t j = 0; j < n; ++j) {
            if (j == (uint32_t)i) continue;
            const SPHParticles::Data pj = particles[j];
            float r = length(pi.position - pj.position);
            if (r < h) pi.density += pj.mass * kernelW(r, h);
        }
    }

    extern "C" GLOBAL void computePressureNew(SPHParticles::Data* particles, uint32_t n, SPHConfig config)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ((uint32_t)i >= n) return;
        SPHParticles::Data& pi = particles[i];

        // Keep a small negative range so particles still pull back together.
        // Fully clamping to zero removes cohesion and makes the fluid drift apart.
        const float pressure = config.stiffness * (pi.density - config.rest_density);
        const float negative_limit = -0.2f * config.stiffness * config.rest_density;
        pi.pressure = fminf(fmaxf(pressure, negative_limit), config.stiffness * config.rest_density);
    }

    extern "C" GLOBAL void applyXsphSmoothingNew(SPHParticles::Data* particles, uint32_t n, SPHConfig config)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ((uint32_t)i >= n) return;

        SPHParticles::Data& pi = particles[i];
        const float h = config.kernel_size;
        const float eps = 0.04f;

        Vec3f delta_v(0.0f);
        for (uint32_t j = 0; j < n; ++j) {
            if (j == (uint32_t)i) continue;
            const SPHParticles::Data pj = particles[j];
            float r = length(pi.position - pj.position);
            if (r < h && pj.density > 1e-8f) {
                float w = kernelW(r, h);
                delta_v += eps * (pj.mass / pj.density) * (pj.velocity - pi.velocity) * w;
            }
        }

        pi.velocity += delta_v;
    }

    extern "C" GLOBAL void computeForcesNew(SPHParticles::Data* particles, uint32_t n, SPHConfig config)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ((uint32_t)i >= n) return;

        SPHParticles::Data& pi = particles[i];
        const float h = config.kernel_size;

        Vec3f f_pressure(0.0f);
        Vec3f f_visc(0.0f);

        for (uint32_t j = 0; j < n; ++j) {
            if (j == (uint32_t)i) continue;
            const SPHParticles::Data pj = particles[j];
            Vec3f r_ij = pj.position - pi.position;
            float r = length(r_ij);
            if (r < 1e-6f || r > h) continue;

            Vec3f dir = r_ij / r;

            // Pressure force: symmetric formulation
            float dWdr = kernelGradScalar(r, h);
            Vec3f pterm = -dir * (pj.mass * (pi.pressure + pj.pressure) * dWdr / (2.0f * pj.density));
            f_pressure += pterm;

            // Viscosity: Laplacian kernel, stabilizing
            float lap = viscosityLaplacian(r, h);
            f_visc += config.viscosity * (pj.mass * (pj.velocity - pi.velocity) * lap) / pj.density;
        }

        // Normalize pressure by local density (avoid divide by zero)
        if (pi.density > 1e-8f) f_pressure *= (1.0f / pi.density);

        // External forces (gravity from config.external_force)
        Vec3f f_ext = Vec3f(config.external_force);

        pi.force = f_pressure + f_visc + f_ext;
    }

    // Symmetric, pairwise collision handled once per pair (i<j)
    extern "C" GLOBAL void particleCollisionNew(SPHParticles::Data* particles, uint32_t n, SPHConfig config)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ((uint32_t)i >= n) return;

        SPHParticles::Data& pi = particles[i];

        const float separation_force = 5.0f;
        const float restitution = 0.15f;
        const float correction_percent = 0.8f;
        const float slop = 0.001f;

        // Each thread processes pairs where i < j
        for (uint32_t j = (uint32_t)i + 1; j < n; ++j) {
            SPHParticles::Data& pj = particles[j];
            Vec3f rij = pi.position - pj.position;
            float dist = length(rij);
            float min_d = pi.radius + pj.radius;
            if (dist < 1e-6f) continue;
            if (dist < min_d) {
                Vec3f nrm = rij / dist;
                Vec3f relv = pi.velocity - pj.velocity;
                float vn = dot(relv, nrm);
                if (vn >= 0.0f) continue; // separating

                float impulse_scalar = -(1.0f + restitution) * vn;
                impulse_scalar /= (1.0f / pi.mass + 1.0f / pj.mass);
                Vec3f impulse = impulse_scalar * nrm;

                Vec3f dv_i = impulse / pi.mass;
                Vec3f dv_j = -impulse / pj.mass;

                // Atomically apply velocity changes
                atomicAddVec3(pi.velocity, dv_i);
                atomicAddVec3(pj.velocity, dv_j);

                // Position correction split between particles
                float overlap = min_d - dist;
                if (overlap > slop) {
                    Vec3f corr = nrm * (overlap * correction_percent / (1.0f / pi.mass + 1.0f / pj.mass));
                    Vec3f corr_i = corr / (2.0f * pi.mass);
                    Vec3f corr_j = -corr / (2.0f * pj.mass);
                    atomicAddVec3(pi.position, corr_i);
                    atomicAddVec3(pj.position, corr_j);
                }

                // Small separation force to reduce clustering
                Vec3f sep = nrm * separation_force * (min_d - dist) / min_d;
                atomicAddVec3(pi.force, sep);
                atomicAddVec3(pj.force, -sep);
            }
        }
    }

    // Wall handling + integrate velocity/position
    extern "C" GLOBAL void updateParticleNew(SPHParticles::Data* particles, uint32_t n, SPHConfig config, AABB wall)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if ((uint32_t)i >= n) return;

        SPHParticles::Data& p = particles[i];
        const float dt = config.time_step;
        const float ks = config.ks;
        const float kd = config.kd;
        const float wall_restitution = 0.35f;

        // Apply wall penalty based on predicted next position
        Vec3f nextPos = p.position + p.velocity * dt;

        // X max
        if (nextPos.x() + p.radius > wall.max().x()) {
            float pen = nextPos.x() + p.radius - wall.max().x();
            float vn = p.velocity.x();
            float f = fmaxf(0.0f, ks * pen - kd * vn);
            p.force += Vec3f(-f, 0.0f, 0.0f);
        }
        // X min
        if (nextPos.x() - p.radius < wall.min().x()) {
            float pen = wall.min().x() - (nextPos.x() - p.radius);
            float vn = p.velocity.x();
            float f = fmaxf(0.0f, ks * pen - kd * vn);
            p.force += Vec3f(f, 0.0f, 0.0f);
        }
        // Y max
        if (nextPos.y() + p.radius > wall.max().y()) {
            float pen = nextPos.y() + p.radius - wall.max().y();
            float vn = p.velocity.y();
            float f = fmaxf(0.0f, ks * pen - kd * vn);
            p.force += Vec3f(0.0f, -f, 0.0f);
        }
        // Y min
        if (nextPos.y() - p.radius < wall.min().y()) {
            float pen = wall.min().y() - (nextPos.y() - p.radius);
            float vn = p.velocity.y();
            float f = fmaxf(0.0f, ks * pen - kd * vn);
            p.force += Vec3f(0.0f, f, 0.0f);
        }
        // Z max
        if (nextPos.z() + p.radius > wall.max().z()) {
            float pen = nextPos.z() + p.radius - wall.max().z();
            float vn = p.velocity.z();
            float f = fmaxf(0.0f, ks * pen - kd * vn);
            p.force += Vec3f(0.0f, 0.0f, -f);
        }
        // Z min
        if (nextPos.z() - p.radius < wall.min().z()) {
            float pen = wall.min().z() - (nextPos.z() - p.radius);
            float vn = p.velocity.z();
            float f = fmaxf(0.0f, ks * pen - kd * vn);
            p.force += Vec3f(0.0f, 0.0f, f);
        }

        // Integrate
        p.velocity += dt * p.force / p.mass;
        nextPos = p.position + dt * p.velocity;

        // Hard clamp + restitution to avoid visible sinking
        if (nextPos.x() + p.radius > wall.max().x()) { nextPos.x() = wall.max().x() - p.radius; if (p.velocity.x() > 0.0f) p.velocity.x() = -p.velocity.x() * wall_restitution; }
        if (nextPos.x() - p.radius < wall.min().x()) { nextPos.x() = wall.min().x() + p.radius; if (p.velocity.x() < 0.0f) p.velocity.x() = -p.velocity.x() * wall_restitution; }
        if (nextPos.y() + p.radius > wall.max().y()) { nextPos.y() = wall.max().y() - p.radius; if (p.velocity.y() > 0.0f) p.velocity.y() = -p.velocity.y() * wall_restitution; }
        if (nextPos.y() - p.radius < wall.min().y()) { nextPos.y() = wall.min().y() + p.radius; if (p.velocity.y() < 0.0f) p.velocity.y() = -p.velocity.y() * wall_restitution; }
        if (nextPos.z() + p.radius > wall.max().z()) { nextPos.z() = wall.max().z() - p.radius; if (p.velocity.z() > 0.0f) p.velocity.z() = -p.velocity.z() * wall_restitution; }
        if (nextPos.z() - p.radius < wall.min().z()) { nextPos.z() = wall.min().z() + p.radius; if (p.velocity.z() < 0.0f) p.velocity.z() = -p.velocity.z() * wall_restitution; }

        p.position = nextPos;
    }

    // Host-entry that launches the new kernel pipeline. Kept separate name to avoid symbol clash.
    extern "C" HOST void solveSPHNew(SPHParticles::Data* d_particles, uint32_t num_particles, SPHConfig config, AABB wall)
    {
        constexpr int NUM_MAX_THREADS = 512;
        const int threads = min((int)num_particles, NUM_MAX_THREADS);
        const int blocks = num_particles / threads + 1;

        dim3 tpb(threads, 1);
        dim3 bpg(blocks, 1);

        computeDensityNew<<<bpg, tpb>>>(d_particles, num_particles, config);
        computePressureNew<<<bpg, tpb>>>(d_particles, num_particles, config);
        computeForcesNew<<<bpg, tpb>>>(d_particles, num_particles, config);
        applyXsphSmoothingNew<<<bpg, tpb>>>(d_particles, num_particles, config);
        particleCollisionNew<<<bpg, tpb>>>(d_particles, num_particles, config);
        updateParticleNew<<<bpg, tpb>>>(d_particles, num_particles, config, wall);
    }

    // Update particle's AABB buffers on device
    extern "C" GLOBAL void updateAABB(const SPHParticles::Data* particles, uint32_t num_particles, OptixAabb* out_aabbs)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data p = particles[idx];

        out_aabbs[idx] = {
            p.position.x() - p.radius,
            p.position.y() - p.radius,
            p.position.z() - p.radius,
            p.position.x() + p.radius,
            p.position.y() + p.radius,
            p.position.z() + p.radius,
        };
    }

    extern "C" HOST void updateParticleAABB(const SPHParticles::Data* particles, uint32_t num_particles, OptixAabb* out_aabbs)
    {
        constexpr int NUM_MAX_THREADS = 1024;
        constexpr int NUM_MAX_BLOCKS = 65536;

        // Determine thread size
        const int num_threads = min((int)num_particles, NUM_MAX_THREADS);
        dim3 threads_per_block(num_threads, 1);

        // Determine block size
        const int num_blocks = num_particles / num_threads + 1;
        dim3 block_dim(num_blocks, 1);
        updateAABB << <block_dim, threads_per_block >> > (particles, num_particles, out_aabbs);
    }

} // namespace prayground
