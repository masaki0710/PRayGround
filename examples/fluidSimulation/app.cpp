#include "app.h"
#include <cstdio>
#include <fstream>
#include <array>
#include <algorithm>
#include <limits>

namespace {
    inline int gridIndex(int x, int y, int z, int nx, int ny)
    {
        return x + nx * (y + ny * z);
    }

    inline Vec3f lerpIso(const Vec3f& p0, const Vec3f& p1, float v0, float v1, float iso)
    {
        const float denom = (v1 - v0);
        float t = (fabsf(denom) < 1e-7f) ? 0.5f : (iso - v0) / denom;
        t = std::clamp(t, 0.0f, 1.0f);
        return p0 + (p1 - p0) * t;
    }

    inline int maxTrianglesForGrid(const int n)
    {
        const int c = n - 1;
        return c * c * c * 12; // 6 tetra / cell * max 2 triangles / tetra
    }

    int extractSurfaceTriangles(
        const std::vector<SPHParticles::Data>& host_particles,
        const AABB& wall,
        int grid_resolution,
        float iso_level,
        float support_radius_scale,
        std::vector<Vec3f>& out_triangle_vertices,
        int max_triangles = std::numeric_limits<int>::max()
    )
    {
        if (host_particles.empty()) {
            out_triangle_vertices.clear();
            return 0;
        }

        const int n = std::max(8, grid_resolution);
        const int nx = n;
        const int ny = n;
        const int nz = n;

        const float radius = host_particles[0].radius;
        const float support = std::max(1e-3f, support_radius_scale * radius);
        const Vec3f minp = wall.min();
        const Vec3f maxp = wall.max();
        const Vec3f extent = maxp - minp;

        const Vec3f step(
            extent.x() / (float)(nx - 1),
            extent.y() / (float)(ny - 1),
            extent.z() / (float)(nz - 1)
        );

        std::vector<float> field(nx * ny * nz, 0.0f);
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const Vec3f p(
                        minp.x() + step.x() * x,
                        minp.y() + step.y() * y,
                        minp.z() + step.z() * z
                    );

                    float phi = 0.0f;
                    for (const auto& pr : host_particles) {
                        const float r = length(p - pr.position);
                        if (r >= support) continue;
                        const float q = 1.0f - (r / support);
                        phi += q * q;
                    }
                    field[gridIndex(x, y, z, nx, ny)] = phi;
                }
            }
        }

        out_triangle_vertices.clear();
        out_triangle_vertices.reserve((size_t)std::min(maxTrianglesForGrid(n), max_triangles) * 3);

        const int tetra[6][4] = {
            {0, 5, 1, 6},
            {0, 1, 2, 6},
            {0, 2, 3, 6},
            {0, 3, 7, 6},
            {0, 7, 4, 6},
            {0, 4, 5, 6}
        };

        int tri_count = 0;
        auto addTri = [&](const Vec3f& a, const Vec3f& b, const Vec3f& c) {
            if (tri_count >= max_triangles)
                return;
            out_triangle_vertices.push_back(a);
            out_triangle_vertices.push_back(b);
            out_triangle_vertices.push_back(c);
            tri_count++;
        };

        for (int z = 0; z < nz - 1; ++z) {
            for (int y = 0; y < ny - 1; ++y) {
                for (int x = 0; x < nx - 1; ++x) {
                    Vec3f cp[8];
                    float cv[8];

                    cp[0] = Vec3f(minp.x() + step.x() * x, minp.y() + step.y() * y, minp.z() + step.z() * z);
                    cp[1] = Vec3f(minp.x() + step.x() * (x + 1), minp.y() + step.y() * y, minp.z() + step.z() * z);
                    cp[2] = Vec3f(minp.x() + step.x() * (x + 1), minp.y() + step.y() * (y + 1), minp.z() + step.z() * z);
                    cp[3] = Vec3f(minp.x() + step.x() * x, minp.y() + step.y() * (y + 1), minp.z() + step.z() * z);
                    cp[4] = Vec3f(minp.x() + step.x() * x, minp.y() + step.y() * y, minp.z() + step.z() * (z + 1));
                    cp[5] = Vec3f(minp.x() + step.x() * (x + 1), minp.y() + step.y() * y, minp.z() + step.z() * (z + 1));
                    cp[6] = Vec3f(minp.x() + step.x() * (x + 1), minp.y() + step.y() * (y + 1), minp.z() + step.z() * (z + 1));
                    cp[7] = Vec3f(minp.x() + step.x() * x, minp.y() + step.y() * (y + 1), minp.z() + step.z() * (z + 1));

                    cv[0] = field[gridIndex(x, y, z, nx, ny)];
                    cv[1] = field[gridIndex(x + 1, y, z, nx, ny)];
                    cv[2] = field[gridIndex(x + 1, y + 1, z, nx, ny)];
                    cv[3] = field[gridIndex(x, y + 1, z, nx, ny)];
                    cv[4] = field[gridIndex(x, y, z + 1, nx, ny)];
                    cv[5] = field[gridIndex(x + 1, y, z + 1, nx, ny)];
                    cv[6] = field[gridIndex(x + 1, y + 1, z + 1, nx, ny)];
                    cv[7] = field[gridIndex(x, y + 1, z + 1, nx, ny)];

                    for (int t = 0; t < 6; ++t) {
                        const int ids[4] = { tetra[t][0], tetra[t][1], tetra[t][2], tetra[t][3] };
                        int inside[4], outside[4];
                        int ni = 0, no = 0;
                        for (int k = 0; k < 4; ++k) {
                            if (cv[ids[k]] >= iso_level) inside[ni++] = ids[k];
                            else outside[no++] = ids[k];
                        }

                        if (ni == 0 || ni == 4) continue;

                        if (ni == 1) {
                            const int a = inside[0], b = outside[0], c = outside[1], d = outside[2];
                            addTri(
                                lerpIso(cp[a], cp[b], cv[a], cv[b], iso_level),
                                lerpIso(cp[a], cp[c], cv[a], cv[c], iso_level),
                                lerpIso(cp[a], cp[d], cv[a], cv[d], iso_level)
                            );
                        }
                        else if (ni == 3) {
                            const int a = outside[0], b = inside[0], c = inside[1], d = inside[2];
                            addTri(
                                lerpIso(cp[a], cp[b], cv[a], cv[b], iso_level),
                                lerpIso(cp[a], cp[d], cv[a], cv[d], iso_level),
                                lerpIso(cp[a], cp[c], cv[a], cv[c], iso_level)
                            );
                        }
                        else {
                            const int a = inside[0], b = inside[1], c = outside[0], d = outside[1];
                            const Vec3f p0 = lerpIso(cp[a], cp[c], cv[a], cv[c], iso_level);
                            const Vec3f p1 = lerpIso(cp[a], cp[d], cv[a], cv[d], iso_level);
                            const Vec3f p2 = lerpIso(cp[b], cp[c], cv[b], cv[c], iso_level);
                            const Vec3f p3 = lerpIso(cp[b], cp[d], cv[b], cv[d], iso_level);
                            addTri(p0, p1, p2);
                            addTri(p1, p3, p2);
                        }

                        if (tri_count >= max_triangles)
                            return tri_count;
                    }
                }
            }
        }

        return tri_count;
    }
}

void App::exportSurfaceOBJ()
{
    const uint32_t num_particles = particles->numPrimitives();
    if (num_particles == 0) {
        printf("[SurfaceExport] No particles to export.\n");
        return;
    }

    std::vector<SPHParticles::Data> host_particles(num_particles);
    CUDA_CHECK(cudaMemcpy(
        host_particles.data(),
        reinterpret_cast<const void*>(particles->devicePtr()),
        sizeof(SPHParticles::Data) * num_particles,
        cudaMemcpyDeviceToHost
    ));

    std::vector<Vec3f> tri_vertices;
    const int tri_count = extractSurfaceTriangles(
        host_particles,
        wall,
        surface_grid_resolution,
        surface_iso_level,
        surface_support_radius_scale,
        tri_vertices
    );

    std::vector<std::array<int, 3>> out_faces;
    out_faces.reserve((size_t)tri_count);

    const std::string outname = "fluid_surface_" + std::to_string(params.frame) + ".obj";
    std::ofstream ofs(outname, std::ios::out | std::ios::trunc);
    if (!ofs) {
        printf("[SurfaceExport] Failed to open output file: %s\n", outname.c_str());
        return;
    }

    for (const auto& v : tri_vertices)
        ofs << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    for (int t = 0; t < tri_count; ++t) {
        const int base = t * 3;
        out_faces.push_back({ base + 1, base + 2, base + 3 });
    }
    for (const auto& f : out_faces)
        ofs << "f " << f[0] << " " << f[1] << " " << f[2] << "\n";

    printf("[SurfaceExport] Exported %s (verts=%zu, tris=%zu, grid=%d)\n",
        outname.c_str(), tri_vertices.size(), out_faces.size(), surface_grid_resolution);
}

void App::buildRealtimeSurfaceMeshBuffers()
{
    const int n = std::max(8, surface_runtime_resolution);
    const int max_tris = maxTrianglesForGrid(n);
    const Vec3f hidden_pos(0.0f, surface_hidden_offset_y, 0.0f);

    surface_runtime_vertices.assign((size_t)max_tris * 3, hidden_pos);
    surface_runtime_faces.clear();
    surface_runtime_faces.reserve((size_t)max_tris);

    for (int i = 0; i < max_tris; ++i) {
        const int base = i * 3;
        surface_runtime_faces.push_back(Face{ Vec3i(base, base + 1, base + 2), Vec3i(0), Vec3i(0) });
    }

    surface_mesh = make_shared<TriangleMesh>(
        surface_runtime_vertices,
        surface_runtime_faces,
        std::vector<Vec3f>(),
        std::vector<Vec2f>()
    );
    surface_mesh->calculateNormalFlat();
    surface_mesh->copyToDevice();
}

void App::updateRealtimeSurfaceMesh()
{
    if (!surface_mesh)
        return;

    const uint32_t num_particles = particles->numPrimitives();
    if (num_particles == 0)
        return;

    std::vector<SPHParticles::Data> host_particles(num_particles);
    CUDA_CHECK(cudaMemcpy(
        host_particles.data(),
        reinterpret_cast<const void*>(particles->devicePtr()),
        sizeof(SPHParticles::Data) * num_particles,
        cudaMemcpyDeviceToHost
    ));

    std::vector<Vec3f> tri_vertices;
    const int max_tris = (int)surface_runtime_faces.size();
    const int tri_count = extractSurfaceTriangles(
        host_particles,
        wall,
        surface_runtime_resolution,
        surface_iso_level,
        surface_support_radius_scale,
        tri_vertices,
        max_tris
    );

    const Vec3f hidden_pos(0.0f, surface_hidden_offset_y, 0.0f);
    std::fill(surface_runtime_vertices.begin(), surface_runtime_vertices.end(), hidden_pos);
    const size_t copy_count = std::min(surface_runtime_vertices.size(), tri_vertices.size());
    std::copy(tri_vertices.begin(), tri_vertices.begin() + copy_count, surface_runtime_vertices.begin());

    auto& vertices = const_cast<std::vector<Vec3f>&>(surface_mesh->vertices());
    if (vertices.size() != surface_runtime_vertices.size())
        return;

    std::copy(surface_runtime_vertices.begin(), surface_runtime_vertices.end(), vertices.begin());
    surface_mesh->calculateNormalFlat();
    surface_mesh->copyToDevice();

    scene.updateObjectGAS("fluidSurface", context, stream);

    if ((params.frame % 30) == 0) {
        printf("[SurfaceMode] updated mesh triangles=%d/%d\n", tri_count, max_tris);
    }
}

void App::rebuildSceneForMode(FluidRenderMode mode)
{
    // Clear all scene objects so the SBT can be rebuilt from a clean cursor.
    scene.clearObjects();

    // Re-add objects in a canonical order.
    scene.addObject("floor", make_shared<Plane>(Vec2f(-75.0f, -75.0f), Vec2f(75.0f, 75.0f)), floor_bsdf, { floor_hitgroup_prg }, Matrix4f::translate(0, -100.0f, 0));

    if (mode == FluidRenderMode::Particles) {
        scene.addObject("particles", particles, particle_bsdf, { particle_hitgroup_prg }, Matrix4f::identity(), {true, true});
    }
    else if (mode == FluidRenderMode::Surface) {
        scene.addObject("particles", particles, particle_bsdf, { particle_hitgroup_prg }, Matrix4f::identity(), {true, true});
        scene.addObject("fluidSurface", surface_mesh, surface_bsdf, { mesh_hitgroup_prg }, Matrix4f::identity(), {true, true});
    }

    // Rebuild acceleration structures
    scene.copyDataToDevice();
    scene.buildAccel(context, stream);
    scene.buildSBT();
    params.handle = scene.accelHandle();
}

void App::initResultBufferOnDevice()
{
    params.frame = 0;
    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    params.result_buffer = (Vec4u*)result_bmp.deviceData();
    params.accum_buffer = (Vec4f*)accum_bmp.deviceData();
}

void App::handleCameraUpdate() {
    if (!is_camera_updated)
        return;
    is_camera_updated = false;

    scene.updateSBT(+(SBTRecordType::Raygen));

    initResultBufferOnDevice();
}

void App::initParticles() {
    float radius = 1.5f;
    uint32_t seed = tea<4>(0, 0);
    std::vector<SPHParticles::Data> particle_data;
    constexpr int NUM_GRID = 25;
    // Center the particle grid at origin
    const float grid_offset = (NUM_GRID - 1) * 4.0f / 2.0f;  // 36.0f
    for (int x = 0; x < NUM_GRID; x++) {
        for (int y = 0; y < NUM_GRID; y++) {
            for (int z = 0; z < NUM_GRID; z++) {
                Vec3f position = Vec3f(x, y, z) * 4.0f - grid_offset;
                Vec3f velocity = Vec3f(0.0f);
                float mass = 1.0f;
                Vec3f perturbation = (UniformSampler::get3D(seed) - 0.5f) * 0.2f;
                position += perturbation;
                auto p = SPHParticles::Data{ position, velocity, mass, radius, 0.0f, 0.0f, Vec3f(0.0f) };
                particle_data.push_back(p);
            }
        }
    }
    particles->setParticles(particle_data);
    particles->copyToDevice();
}

void App::applyFluidPreset()
{
    sph_config = {
        .kernel_size = 7.0f,
        .rest_density = 0.12f,
        .external_force = make_float3(0.0f, -9.8f, 0.0f),
        .time_step = 0.008f,
        .stiffness = 8.0f,
        .viscosity = 0.02f,
        .ks = 25.0f,
        .kd = 2.0f
    };
    params.sph_config = sph_config;
}

// ------------------------------------------------------------------
void App::setup()
{
    stream = 0;
    CUDA_CHECK(cudaFree(0));

    // Initialize context
    OPTIX_CHECK(optixInit());
    context.disableValidation();
    context.create();

    // Initialize pipeline
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(5);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);

    // Create module
    Module module = pipeline.createModuleFromCudaFile(context, "kernels.cu");

    const int width = pgGetWidth();
    const int height = pgGetHeight();
    result_bmp.allocate(PixelFormat::RGBA, width, height);
    result_bmp.allocateDevicePtr();
    accum_bmp.allocate(PixelFormat::RGBA, width, height);
    accum_bmp.allocateDevicePtr();

    // Configuration of launch parameters
    params.width = width;
    params.height = height;
    params.samples_per_launch = 8;
    params.frame = 0;
    params.max_depth = 8;
    params.result_buffer = (Vec4u*)result_bmp.deviceData();
    params.accum_buffer = (Vec4f*)accum_bmp.deviceData();

    scene.setup({ true, true });

    // Camera settings
    std::shared_ptr<Camera> camera = make_shared<Camera>();
    camera->setOrigin(300, 200, 300);
    camera->setLookat(0, 0, 0);
    camera->setUp(0, 1, 0);
    camera->setFov(40);
    camera->setAspect((float)width / height);
    camera->enableTracking(pgGetCurrentWindow());
    scene.setCamera(camera);

    // Raygen program
    ProgramGroup raygen = pipeline.createRaygenProgram(context, module, "__raygen__pinhole");
    scene.bindRaygenProgram(raygen);

    struct Callable {
        Callable(const std::pair<ProgramGroup, uint32_t>& callable)
            : program(callable.first), ID(callable.second) {}
        ProgramGroup program;
        uint32_t ID;
    };

    auto setupCallable = [&](const string& dc_name, const string& cc_name) {
        Callable callable = pipeline.createCallablesProgram(context, module, dc_name, cc_name);
        scene.bindCallablesProgram(callable.program);
        return callable.ID;
    };

    // Texture programs
    auto bitmap_id = setupCallable("__direct_callable__bitmap", "");
    auto checker_id = setupCallable("__direct_callable__checker", "");
    auto constant_id = setupCallable("__direct_callable__constant", "");

    // Miss program
    ProgramGroup miss = pipeline.createMissProgram(context, module, "__miss__envmap");
    scene.bindMissPrograms({miss});
    auto envmap_texture = make_shared<FloatBitmapTexture>("resources/image/drackenstein_quarry_4k.exr", bitmap_id);
    scene.setEnvmap(envmap_texture);

    // Hitgroup program
    mesh_hitgroup_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");
    ProgramGroup plane_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__plane", "__intersection__plane");
    particle_hitgroup_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__sphere", "__intersection__particle");

    // Surface programs
    SurfaceCallableID diffuse_id = {
        .sample = setupCallable("__direct_callable__sample_diffuse", ""),
        .bsdf = setupCallable("__direct_callable__bsdf_diffuse", ""),
        .pdf = setupCallable("__direct_callable__pdf_diffuse", "")
    };
    SurfaceCallableID dielectric_id = {
        .sample = setupCallable("__direct_callable__sample_dielectric", ""),
        .bsdf = setupCallable("__direct_callable__bsdf_dielectric", ""),
        .pdf = setupCallable("__direct_callable__pdf_dielectric", "")
    };
    SurfaceCallableID conductor_id = {
        .sample = setupCallable("__direct_callable__sample_conductor", ""),
        .bsdf = setupCallable("__direct_callable__bsdf_conductor", ""),
        .pdf = setupCallable("__direct_callable__pdf_conductor", "")
    };
    auto area_emitter_callable_id = setupCallable("__direct_callable__area_emitter", "");
    SurfaceCallableID area_emitter_id = {
        .sample = area_emitter_callable_id,
        .bsdf = area_emitter_callable_id,
        .pdf = area_emitter_callable_id
    };

    // Create surfaces
    floor_bsdf = make_shared<Diffuse>(diffuse_id, make_shared<CheckerTexture>(Vec3f(0.2f), Vec3f(0.8f), 10, checker_id));
    particle_bsdf = make_shared<Dielectric>(dielectric_id, make_shared<ConstantTexture>(Vec3f(0.8f), constant_id), 1.5f);
    surface_bsdf = make_shared<Dielectric>(dielectric_id, make_shared<ConstantTexture>(Vec3f(0.65f, 0.8f, 0.95f), constant_id), 1.33f);

    // Initialize fluid particles
    particles = make_shared<SPHParticles>();
    initParticles();

    // Fluid particle
    buildRealtimeSurfaceMeshBuffers();

    floor_hitgroup_prg = plane_prg;

    // Floor
    Vec3f wall_min(-75.0f, -100.0f, -75.0f);
    Vec3f wall_max(75.0f, 500.0f, 75.0f);

    CUDA_CHECK(cudaStreamCreate(&stream));


    // Configuration of SPH parameter
    applyFluidPreset();

    wall = AABB(wall_min, wall_max);

    rebuildSceneForMode(render_mode);
    if (render_mode == FluidRenderMode::Surface)
        updateRealtimeSurfaceMesh();
    pipeline.create(context);

    // GUI setting
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    const char* glsl_version = "#version 150";
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(pgGetCurrentWindow()->windowPtr(), true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();
    initResultBufferOnDevice();

    scene.launchRay(context, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    result_bmp.copyFromDevice();

    params.frame++;

    solveSPHNew((SPHParticles::Data*)particles->devicePtr(), particles->numPrimitives(), params.sph_config, wall);

    if (request_surface_export) {
        request_surface_export = false;
        exportSurfaceOBJ();
    }

    // Check for render mode change
    if (requested_mode != render_mode) {
        render_mode = requested_mode;
        rebuildSceneForMode(render_mode);
    }

    scene.updateObjectGAS("particles", context, stream);
    if (render_mode == FluidRenderMode::Surface)
        updateRealtimeSurfaceMesh();
    
    // Update params with current render mode for shader
    params.render_mode = (render_mode == FluidRenderMode::Particles) ? 0 : 1;

    scene.updateAccel(context, stream);
}

// ------------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Fluid Simulation");

    ImGui::SliderFloat("Kernel Size", &sph_config.kernel_size, 1.0f, 100.0f);
    ImGui::SliderFloat("Rest Density", &sph_config.rest_density, 0.1f, 100.0f);
    ImGui::SliderFloat("Time Step", &sph_config.time_step, 0.001f, 0.1f);
    ImGui::SliderFloat("Stiffness", &sph_config.stiffness, 0.0f, 10.0f);
    ImGui::SliderFloat("Viscosity", &sph_config.viscosity, 0.0f, 1.0f);
    ImGui::SliderFloat("Ks", &sph_config.ks, 1.0f, 50.0f);
    ImGui::SliderFloat("Kd", &sph_config.kd, 1.0f, 50.0f);
    params.sph_config = sph_config;

    int render_mode_index = (render_mode == FluidRenderMode::Particles) ? 0 : 1;
    if (ImGui::Combo("Render Mode", &render_mode_index, "Particle Rendering\0Surface Rendering\0")) {
        // Removed redundant render_mode assignment
        
        requested_mode = (render_mode_index == 0) ? FluidRenderMode::Particles : FluidRenderMode::Surface;
    }

    if (render_mode == FluidRenderMode::Surface) {
        ImGui::Text("Realtime surface grid: %d^3 (fixed topology)", surface_runtime_resolution);
    }

    ImGui::Separator();
    ImGui::Text("Surface Export (Marching Tetrahedra)");
    ImGui::SliderInt("Grid Resolution", &surface_grid_resolution, 16, 48);
    ImGui::SliderFloat("Iso Level", &surface_iso_level, 0.05f, 1.5f);
    ImGui::SliderFloat("Support Radius Scale", &surface_support_radius_scale, 1.2f, 3.5f);
    if (ImGui::Button("Export Fluid Surface OBJ")) {
        request_surface_export = true;
    }

    ImGui::Text("Camera info:");
    auto camera = scene.camera();
    ImGui::Text("Origin: %f %f %f", camera->origin().x(), camera->origin().y(), camera->origin().z());
    ImGui::Text("Lookat: %f %f %f", camera->lookat().x(), camera->lookat().y(), camera->lookat().z());

    if (ImGui::Button("Reset")) {
        initParticles();
    }

    ImGui::SameLine();
    if (ImGui::Button("Apply Fluid Preset")) {
        applyFluidPreset();
    }

    ImGui::Separator();
    ImGui::Checkbox("Debug Logging", &debug_logging_enabled);

    if (debug_logging_enabled) {
        ImGui::Text("Debug Info (Frame %d):", debug_frame_count);
        ImGui::Text("Config: gravity=%.2f, ks=%.1f, kd=%.1f", 
                   sph_config.external_force.y, sph_config.ks, sph_config.kd);
    }

    ImGui::End();
    ImGui::Render();

    result_bmp.draw(0, 0);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// ------------------------------------------------------------------
void App::mousePressed(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    if (button == MouseButton::Middle) is_camera_updated = true;
}

// ------------------------------------------------------------------
void App::mouseReleased(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseMoved(float x, float y)
{
    
}

// ------------------------------------------------------------------
void App::mouseScrolled(float x, float y)
{
    is_camera_updated = true;
}

// ------------------------------------------------------------------
void App::keyPressed(int key)
{
    if (key == 'M' || key == 'm') {
        request_surface_export = true;
    }

    if (key == 'R' || key == 'r') {
        requested_mode = (render_mode == FluidRenderMode::Particles) ? FluidRenderMode::Surface : FluidRenderMode::Particles;
    }
}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{

}



