#pragma once

#include <prayground/prayground.h>
#include "params.h"

#include <prayground/ext/imgui/imgui.h>
#include <prayground/ext/imgui/imgui_impl_glfw.h>
#include <prayground/ext/imgui/imgui_impl_opengl3.h>

using namespace std;

class App : public BaseApp 
{
public:
    void setup();
    void update();
    void draw();

    void mousePressed(float x, float y, int button);
    void mouseDragged(float x, float y, int button);
    void mouseReleased(float x, float y, int button);
    void mouseMoved(float x, float y);
    void mouseScrolled(float xoffset, float yoffset);

    void keyPressed(int key);
    void keyReleased(int key);
private:
    enum class FluidRenderMode {
        Particles = 0,
        Surface = 1
    };

    void initResultBufferOnDevice();
    void handleCameraUpdate();
    void initParticles();
    void applyFluidPreset();
    void exportSurfaceOBJ();
    void buildRealtimeSurfaceMeshBuffers();
    void updateRealtimeSurfaceMesh();
    void rebuildSceneForMode(FluidRenderMode mode);

    Context context;
    CUstream stream;
    Pipeline pipeline;
    
    LaunchParams params;

    Bitmap result_bmp;
    FloatBitmap accum_bmp;

    static constexpr uint32_t NRay = 1;
    using AppScene = Scene<Camera, NRay>;
    AppScene scene;

    bool is_camera_updated;

    SPHConfig sph_config;
    shared_ptr<SPHParticles> particles;
    shared_ptr<TriangleMesh> surface_mesh;
    shared_ptr<Material> floor_bsdf;
    shared_ptr<Material> particle_bsdf;
    shared_ptr<Material> surface_bsdf;
    ProgramGroup mesh_hitgroup_prg;
    ProgramGroup floor_hitgroup_prg;
    ProgramGroup particle_hitgroup_prg;

    AABB wall;

    // Debug logging
    int debug_frame_count;
    bool debug_logging_enabled;
    std::vector<int> watched_particle_ids;

    // Surface export controls (particle -> mesh)
    bool request_surface_export{ false };
    int surface_grid_resolution{ 28 };
    float surface_iso_level{ 0.35f };
    float surface_support_radius_scale{ 2.2f };

    // Real-time surface mode (fixed topology, per-frame vertex update)
    FluidRenderMode render_mode{ FluidRenderMode::Particles };
        FluidRenderMode requested_mode{ FluidRenderMode::Particles };
    int surface_runtime_resolution{ 12 };
    float surface_hidden_offset_y{ -100000.0f };
    std::vector<Vec3f> surface_runtime_vertices;
    std::vector<Face> surface_runtime_faces;
};