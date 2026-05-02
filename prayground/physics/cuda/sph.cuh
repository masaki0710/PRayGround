#include <prayground/physics/sph.h>

namespace prayground {
    // Entry point for SPH simulation on CUDA
    extern "C" HOST void solveSPH(
        SPHParticles::Data* d_particles,   // Device pointer to particles
        uint32_t num_particles, 
        SPHConfig config,                 // Changed to reference for adaptive timestep modification
        AABB wall = AABB(Vec3f(-10000), Vec3f(10000))
    );

    // New pipeline entry for alternative SPH implementation
    extern "C" HOST void solveSPHNew(
        SPHParticles::Data* d_particles,
        uint32_t num_particles,
        SPHConfig config,
        AABB wall = AABB(Vec3f(-10000), Vec3f(10000))
    );

    extern "C" HOST void solveSPHDebug(
        SPHParticles::Data* d_particles,
        uint32_t num_particles,
        SPHConfig config,
        AABB wall,
        const int* watched_particle_ids,
        uint32_t watched_particle_count,
        int frame_index
    );

    extern "C" HOST void updateParticleAABB(
        const SPHParticles::Data * particles,
        uint32_t num_particles,
        OptixAabb * out_aabbs
    );
} // namespace prayground