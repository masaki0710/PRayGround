// Smoothed Particle Hydrodynamics 

#include <prayground/physics/cuda/sph.cuh>
#include <prayground/math/util.h>
#include <vector>
#include <stdio.h>

namespace prayground {

    DEVICE float cubicSpline(float q)
    {
        if ((0.0f <= q) && (q <= 0.5f))
            return 6.0f * (pow3(q) - pow2(q)) + 1.0f;
        else if ((0.5f < q) && (q <= 1.0f))
            return 2.0f * pow3(1.0f - q);
        else
            return 0.0f;
    }

    DEVICE float cubicSplineDerivative(float q)
    {
        if ((0.0f <= q) && (q <= 0.5f))
            return 6.0f * (3.0f * pow2(q) - 2.0f * q);
        else if ((0.5f < q) && (q <= 1.0f))
            return -6.0f * pow2(1.0f - q);
        else {
            return 0.0f;
        }
    }

    DEVICE float particleKernel(float r, float kernel_size)
    {
        auto q = r / kernel_size;
        auto norm_factor = 8.0f / (math::pi * pow3(kernel_size));
        return norm_factor * cubicSpline(q);
    }

    DEVICE float particleKernelDerivative(float r, float kernel_size)
    {
        auto q = r / kernel_size;
        auto norm_factor = 8.0f / (math::pi * pow3(kernel_size));
        return norm_factor * cubicSplineDerivative(q);
    }

    DEVICE float particleViscosityLaplacian(float r, float kernel_size)
    {
        if (r < 0.0f || r > kernel_size)
            return 0.0f;

        // Stable 3D viscosity kernel laplacian.
        // Positive values make the force smooth relative velocity instead of amplifying it.
        const float h3 = pow3(kernel_size);
        return 45.0f / (math::pi * h3 * h3) * (kernel_size - r);
    }

    // Atomic add helper for Vec3f without using device lambdas (avoids requiring C++14 device lambda support)
    DEVICE inline void atomicAddVec3(Vec3f& dst, const Vec3f& v)
    {
        atomicAdd(&dst.x(), v.x());
        atomicAdd(&dst.y(), v.y());
        atomicAdd(&dst.z(), v.z());
    }

    extern "C" GLOBAL void computeDensity(SPHParticles::Data* particles, uint32_t num_particles, SPHConfig config)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        const float h = config.kernel_size;
        SPHParticles::Data& pi = particles[idx];
        pi.density = pi.mass * particleKernel(0.0f, h);

        for (auto j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            // Reconstruct density from mass and kernel
            auto pj = particles[j];
            float r = length(pi.position - pj.position);

            // Ignore particles outside of kernel size
            if (r < h) {
                pi.density += pj.mass * particleKernel(r, h);
            }
        }
    }

    extern "C" GLOBAL void computePressure(SPHParticles::Data* particles, uint32_t num_particles, SPHConfig config)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];
        pi.pressure = config.stiffness * (pi.density - config.rest_density);
    }

    extern "C" GLOBAL void computeForce(SPHParticles::Data* particles, uint32_t num_particles, SPHConfig config)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];

        Vec3f pressure_force(0.0f);
        Vec3f viscosity_force(0.0f);

        const float h = config.kernel_size;

        for (auto j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            auto pj = particles[j];

            Vec3f pi2pj = pj.position - pi.position;
            float r = length(pi2pj);
            if (r > 1e-6f && r < h) {
                auto dir = pi2pj / r;
                auto viscosity_laplacian = particleViscosityLaplacian(r, h);
                auto kernel_derivative = particleKernelDerivative(r, h);
                viscosity_force += config.viscosity * (pj.mass * (pj.velocity - pi.velocity) * viscosity_laplacian) / pj.density;

                pressure_force += -dir * (pj.mass * (pj.pressure + pi.pressure)) * kernel_derivative / (2.0f * pj.density);
            }
        }

        pressure_force *= -1.0f / pi.density;

        pi.force = pressure_force + viscosity_force + Vec3f(config.external_force);
    }

    // Original collision with asymmetric position correction (causes x+ drift)
    GLOBAL void particleCollision(SPHParticles::Data* particles, uint32_t num_particles, SPHConfig config)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];
        
        // Collision parameters
        const float collision_damping = 0.8f;  // Energy loss during collision
        const float separation_force = 10.0f;  // Force to separate overlapping particles
        
        for (uint32_t j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            SPHParticles::Data& pj = particles[j];
            
            Vec3f relative_pos = pi.position - pj.position;
            float distance = length(relative_pos);
            float min_distance = pi.radius + pj.radius;
            
            // Check for collision/overlap
            if (distance < min_distance && distance > 1e-6f) {
                // Normalize the collision direction
                Vec3f collision_normal = relative_pos / distance;
                
                // Calculate relative velocity
                Vec3f relative_velocity = pi.velocity - pj.velocity;
                float velocity_along_normal = dot(relative_velocity, collision_normal);
                
                // Don't resolve if velocities are separating
                if (velocity_along_normal > 0) continue;
                
                // Calculate restitution (bounciness)
                float restitution = 0.3f; // Moderate bounce for fluid particles
                
                // Calculate impulse scalar
                float impulse_scalar = -(1 + restitution) * velocity_along_normal;
                impulse_scalar /= (1.0f / pi.mass + 1.0f / pj.mass);
                
                // Apply impulse to separate particles
                Vec3f impulse = impulse_scalar * collision_normal;
                pi.velocity += impulse / pi.mass * collision_damping;
                
                // Position correction to prevent sinking
                float overlap = min_distance - distance;
                float correction_percent = 0.8f; // Percentage of overlap to correct
                float slop = 0.01f; // Small allowable overlap to prevent jitter
                
                if (overlap > slop) {
                    Vec3f correction = collision_normal * (overlap * correction_percent / (1.0f / pi.mass + 1.0f / pj.mass));
                    pi.position += correction / pi.mass;
                }
                
                // Add small separation force to prevent clustering
                Vec3f separation = collision_normal * separation_force * (min_distance - distance) / min_distance;
                pi.force += separation;
            }
        }
    }

    // Fixed collision: position correction applied only when BOTH particles converge (reduce correction factor)
    GLOBAL void particleCollisionFixed(SPHParticles::Data* particles, uint32_t num_particles, SPHConfig config)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];
        
        const float collision_damping = 0.25f;
        const float separation_force = 1.5f;
        
        for (uint32_t j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            SPHParticles::Data& pj = particles[j];
            
            Vec3f relative_pos = pi.position - pj.position;
            float distance = length(relative_pos);
            float min_distance = pi.radius + pj.radius;
            
            if (distance < min_distance && distance > 1e-6f) {
                Vec3f collision_normal = relative_pos / distance;
                Vec3f relative_velocity = pi.velocity - pj.velocity;
                float velocity_along_normal = dot(relative_velocity, collision_normal);
                
                if (velocity_along_normal > 0) continue;
                
                float restitution = 0.3f;
                float impulse_scalar = -(1 + restitution) * velocity_along_normal;
                impulse_scalar /= (1.0f / pi.mass + 1.0f / pj.mass);
                
                Vec3f impulse = impulse_scalar * collision_normal;
                pi.velocity += impulse / pi.mass * collision_damping;
                
                // TEST: Position correction ONLY (no separation force)
                float overlap = min_distance - distance;
                float correction_percent = 0.8f;
                float slop = 0.01f;
                
                if (overlap > slop) {
                    Vec3f correction = collision_normal * (overlap * correction_percent / (1.0f / pi.mass + 1.0f / pj.mass));
                    pi.position += correction / pi.mass;
                }
                
                // Separation force DISABLED
                // Vec3f separation = collision_normal * separation_force * (min_distance - distance) / min_distance;
                // pi.force += separation;
            }
        }
    }

    // Fixed collision: Process each pair only once (i < j) to eliminate loop-order bias
    GLOBAL void particleCollisionFixed_Symmetric(SPHParticles::Data* particles, uint32_t num_particles, SPHConfig config)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];
        
        const float collision_damping = 0.8f;
        const float separation_force = 10.0f;
        
        // CRITICAL FIX: Only process pairs where i < j (each pair processed exactly once)
        // Use atomic updates to avoid races when multiple threads modify the same particle

        for (uint32_t j = idx + 1; j < num_particles; j++) {
            SPHParticles::Data& pj = particles[j];
            
            Vec3f relative_pos = pi.position - pj.position;
            float distance = length(relative_pos);
            float min_distance = pi.radius + pj.radius;
            
            if (distance < min_distance && distance > 1e-6f) {
                Vec3f collision_normal = relative_pos / distance;
                Vec3f relative_velocity = pi.velocity - pj.velocity;
                float velocity_along_normal = dot(relative_velocity, collision_normal);
                
                // Only resolve if particles are converging (relative velocity < 0 along normal)
                if (velocity_along_normal > 0) continue;
                
                float restitution = 0.3f;
                float impulse_scalar = -(1 + restitution) * velocity_along_normal;
                impulse_scalar /= (1.0f / pi.mass + 1.0f / pj.mass);
                
                // Apply impulse to both particles symmetrically using atomic adds
                Vec3f impulse = impulse_scalar * collision_normal;
                Vec3f dv_i = impulse / pi.mass * collision_damping;
                Vec3f dv_j = -impulse / pj.mass * collision_damping;

                // atomic add velocity deltas
                atomicAddVec3(pi.velocity, dv_i);
                atomicAddVec3(pj.velocity, dv_j);

                // Position correction for both particles (atomic)
                float overlap = min_distance - distance;
                float correction_percent = 0.8f;
                float slop = 0.01f;

                if (overlap > slop) {
                    Vec3f correction = collision_normal * (overlap * correction_percent / (1.0f / pi.mass + 1.0f / pj.mass));
                    Vec3f corr_i = correction / (2.0f * pi.mass);
                    Vec3f corr_j = -correction / (2.0f * pj.mass);
                    atomicAddVec3(pi.position, corr_i);
                    atomicAddVec3(pj.position, corr_j);
                }

                // Separation force (applied to both via atomic adds)
                Vec3f separation = collision_normal * separation_force * (min_distance - distance) / min_distance;
                atomicAddVec3(pi.force, separation);
                atomicAddVec3(pj.force, -separation);
            }
        }
    }

    // Original collision function restored for comparison
    GLOBAL void particleCollisionOriginal(SPHParticles::Data* particles, uint32_t num_particles, SPHConfig config)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];
        
        const float collision_damping = 0.8f;
        const float separation_force = 10.0f;
        
        for (uint32_t j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            SPHParticles::Data& pj = particles[j];
            
            Vec3f relative_pos = pi.position - pj.position;
            float distance = length(relative_pos);
            float min_distance = pi.radius + pj.radius;
            
            if (distance < min_distance && distance > 1e-6f) {
                Vec3f collision_normal = relative_pos / distance;
                Vec3f relative_velocity = pi.velocity - pj.velocity;
                float velocity_along_normal = dot(relative_velocity, collision_normal);
                
                if (velocity_along_normal > 0) continue;
                
                float restitution = 0.3f;
                float impulse_scalar = -(1 + restitution) * velocity_along_normal;
                impulse_scalar /= (1.0f / pi.mass + 1.0f / pj.mass);
                
                Vec3f impulse = impulse_scalar * collision_normal;
                pi.velocity += impulse / pi.mass * collision_damping;
                
                // Position correction
                float overlap = min_distance - distance;
                float correction_percent = 0.8f;
                float slop = 0.01f;
                
                if (overlap > slop) {
                    Vec3f correction = collision_normal * (overlap * correction_percent / (1.0f / pi.mass + 1.0f / pj.mass));
                    pi.position += correction / pi.mass;
                }
                
                // Separation force
                Vec3f separation = collision_normal * separation_force * (min_distance - distance) / min_distance;
                pi.force += separation;
            }
        }
    }

        extern "C" GLOBAL void updateParticle(SPHParticles::Data* particles, uint32_t num_particles, SPHConfig config, AABB wall)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];

        const float kd = config.kd;
        const float ks = config.ks;
        const float wall_restitution = 0.2f;

        const float next_x = pi.position.x() + pi.velocity.x() * config.time_step;
        const float next_y = pi.position.y() + pi.velocity.y() * config.time_step;
        const float next_z = pi.position.z() + pi.velocity.z() * config.time_step;

        const Vec3f normal_x_max(-1.0f, 0.0f, 0.0f);
        const Vec3f normal_x_min(1.0f, 0.0f, 0.0f);
        const Vec3f normal_y_max(0.0f, -1.0f, 0.0f);
        const Vec3f normal_y_min(0.0f, 1.0f, 0.0f);
        const Vec3f normal_z_max(0.0f, 0.0f, -1.0f);
        const Vec3f normal_z_min(0.0f, 0.0f, 1.0f);

        if (next_x + pi.radius > wall.max().x()) {
            const float penetration = next_x + pi.radius - wall.max().x();
            const float vn = dot(pi.velocity, normal_x_max);
            const float force_mag = fmaxf(0.0f, ks * penetration - kd * vn);
            pi.force += normal_x_max * force_mag;
        }

        if (next_x - pi.radius < wall.min().x()) {
            const float penetration = wall.min().x() - (next_x - pi.radius);
            const float vn = dot(pi.velocity, normal_x_min);
            const float force_mag = fmaxf(0.0f, ks * penetration - kd * vn);
            pi.force += normal_x_min * force_mag;
        }

        if (next_y + pi.radius > wall.max().y()) {
            const float penetration = next_y + pi.radius - wall.max().y();
            const float vn = dot(pi.velocity, normal_y_max);
            const float force_mag = fmaxf(0.0f, ks * penetration - kd * vn);
            pi.force += normal_y_max * force_mag;
        }

        if (next_y - pi.radius < wall.min().y()) {
            const float penetration = wall.min().y() - (next_y - pi.radius);
            const float vn = dot(pi.velocity, normal_y_min);
            const float force_mag = fmaxf(0.0f, ks * penetration - kd * vn);
            pi.force += normal_y_min * force_mag;
        }

        if (next_z + pi.radius > wall.max().z()) {
            const float penetration = next_z + pi.radius - wall.max().z();
            const float vn = dot(pi.velocity, normal_z_max);
            const float force_mag = fmaxf(0.0f, ks * penetration - kd * vn);
            pi.force += normal_z_max * force_mag;
        }

        if (next_z - pi.radius < wall.min().z()) {
            const float penetration = wall.min().z() - (next_z - pi.radius);
            const float vn = dot(pi.velocity, normal_z_min);
            const float force_mag = fmaxf(0.0f, ks * penetration - kd * vn);
            pi.force += normal_z_min * force_mag;
        }

        // Update velocity
        pi.velocity += config.time_step * pi.force / pi.mass;

        // Update position with hard wall clamping so the particle does not visibly sink into the wall
        Vec3f next_pos = pi.position + config.time_step * pi.velocity;

        if (next_pos.x() + pi.radius > wall.max().x()) {
            next_pos.x() = wall.max().x() - pi.radius;
            if (pi.velocity.x() > 0.0f) pi.velocity.x() = -pi.velocity.x() * wall_restitution;
        }
        if (next_pos.x() - pi.radius < wall.min().x()) {
            next_pos.x() = wall.min().x() + pi.radius;
            if (pi.velocity.x() < 0.0f) pi.velocity.x() = -pi.velocity.x() * wall_restitution;
        }

        if (next_pos.y() + pi.radius > wall.max().y()) {
            next_pos.y() = wall.max().y() - pi.radius;
            if (pi.velocity.y() > 0.0f) pi.velocity.y() = -pi.velocity.y() * wall_restitution;
        }
        if (next_pos.y() - pi.radius < wall.min().y()) {
            next_pos.y() = wall.min().y() + pi.radius;
            if (pi.velocity.y() < 0.0f) pi.velocity.y() = -pi.velocity.y() * wall_restitution;
        }

        if (next_pos.z() + pi.radius > wall.max().z()) {
            next_pos.z() = wall.max().z() - pi.radius;
            if (pi.velocity.z() > 0.0f) pi.velocity.z() = -pi.velocity.z() * wall_restitution;
        }
        if (next_pos.z() - pi.radius < wall.min().z()) {
            next_pos.z() = wall.min().z() + pi.radius;
            if (pi.velocity.z() < 0.0f) pi.velocity.z() = -pi.velocity.z() * wall_restitution;
        }

        pi.position = next_pos;
    }

    extern "C" HOST void solveSPH(SPHParticles::Data* d_particles, uint32_t num_particles, SPHConfig config, AABB wall)
    {
        constexpr int NUM_MAX_THREADS = 1024;
        constexpr int NUM_MAX_BLOCKS = 65536;

        // Determine thread size
        const int num_threads = min((int)num_particles, NUM_MAX_THREADS);
        dim3 threads_per_block(num_threads, 1);

        // Determine block size
        const int num_blocks = num_particles / num_threads + 1;
        dim3 block_dim(num_blocks, 1);

        // SPH fluid dynamics
        computeDensity << <block_dim, threads_per_block >> > (d_particles, num_particles, config);
        computePressure << <block_dim, threads_per_block >> > (d_particles, num_particles, config);
        computeForce << <block_dim, threads_per_block >> > (d_particles, num_particles, config);
        
        // FIXED: Symmetric collision response with i < j to eliminate loop-order bias
        // particleCollisionFixed_Symmetric << <block_dim, threads_per_block >> > (d_particles, num_particles, config);
        
        // Update positions and handle wall collisions
        updateParticle << <block_dim, threads_per_block >> > (d_particles, num_particles, config, wall);
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