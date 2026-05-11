#include "app.h"
#include <cstdio>
#include <fstream>
#include <array>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <prayground/physics/cuda/surface_field.cuh>
#include <prayground/physics/cuda/surface_mesh.cuh>
#include <prayground/shape/cuda/trianglemesh_normals.cuh>

namespace marching_cubes_tables {
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
        return c * c * c * 5; // Marching cubes: max 5 triangles per cell
    }

    extern const int kEdgeTable[256] = {
        0x0000, 0x0109, 0x0203, 0x030a, 0x0406, 0x050f, 0x0605, 0x070c,
        0x080c, 0x0905, 0x0a0f, 0x0b06, 0x0c0a, 0x0d03, 0x0e09, 0x0f00,
        0x0190, 0x0099, 0x0393, 0x029a, 0x0596, 0x049f, 0x0795, 0x069c,
        0x099c, 0x0895, 0x0b9f, 0x0a96, 0x0d9a, 0x0c93, 0x0f99, 0x0e90,
        0x0230, 0x0339, 0x0033, 0x013a, 0x0636, 0x073f, 0x0435, 0x053c,
        0x0a3c, 0x0b35, 0x083f, 0x0936, 0x0e3a, 0x0f33, 0x0c39, 0x0d30,
        0x03a0, 0x02a9, 0x01a3, 0x00aa, 0x07a6, 0x06af, 0x05a5, 0x04ac,
        0x0bac, 0x0aa5, 0x09af, 0x08a6, 0x0faa, 0x0ea3, 0x0da9, 0x0ca0,
        0x0460, 0x0569, 0x0663, 0x076a, 0x0066, 0x016f, 0x0265, 0x036c,
        0x0c6c, 0x0d65, 0x0e6f, 0x0f66, 0x086a, 0x0963, 0x0a69, 0x0b60,
        0x05f0, 0x04f9, 0x07f3, 0x06fa, 0x01f6, 0x00ff, 0x03f5, 0x02fc,
        0x0dfc, 0x0cf5, 0x0fff, 0x0ef6, 0x09fa, 0x08f3, 0x0bf9, 0x0af0,
        0x0650, 0x0759, 0x0453, 0x055a, 0x0256, 0x035f, 0x0055, 0x015c,
        0x0e5c, 0x0f55, 0x0c5f, 0x0d56, 0x0a5a, 0x0b53, 0x0859, 0x0950,
        0x07c0, 0x06c9, 0x05c3, 0x04ca, 0x03c6, 0x02cf, 0x01c5, 0x00cc,
        0x0fcc, 0x0ec5, 0x0dcf, 0x0cc6, 0x0bca, 0x0ac3, 0x09c9, 0x08c0,
        0x08c0, 0x09c9, 0x0ac3, 0x0bca, 0x0cc6, 0x0dcf, 0x0ec5, 0x0fcc,
        0x00cc, 0x01c5, 0x02cf, 0x03c6, 0x04ca, 0x05c3, 0x06c9, 0x07c0,
        0x0950, 0x0859, 0x0b53, 0x0a5a, 0x0d56, 0x0c5f, 0x0f55, 0x0e5c,
        0x015c, 0x0055, 0x035f, 0x0256, 0x055a, 0x0453, 0x0759, 0x0650,
        0x0af0, 0x0bf9, 0x08f3, 0x09fa, 0x0ef6, 0x0fff, 0x0cf5, 0x0dfc,
        0x02fc, 0x03f5, 0x00ff, 0x01f6, 0x06fa, 0x07f3, 0x04f9, 0x05f0,
        0x0b60, 0x0a69, 0x0963, 0x086a, 0x0f66, 0x0e6f, 0x0d65, 0x0c6c,
        0x036c, 0x0265, 0x016f, 0x0066, 0x076a, 0x0663, 0x0569, 0x0460,
        0x0ca0, 0x0da9, 0x0ea3, 0x0faa, 0x08a6, 0x09af, 0x0aa5, 0x0bac,
        0x04ac, 0x05a5, 0x06af, 0x07a6, 0x00aa, 0x01a3, 0x02a9, 0x03a0,
        0x0d30, 0x0c39, 0x0f33, 0x0e3a, 0x0936, 0x083f, 0x0b35, 0x0a3c,
        0x053c, 0x0435, 0x073f, 0x0636, 0x013a, 0x0033, 0x0339, 0x0230,
        0x0e90, 0x0f99, 0x0c93, 0x0d9a, 0x0a96, 0x0b9f, 0x0895, 0x099c,
        0x069c, 0x0795, 0x049f, 0x0596, 0x029a, 0x0393, 0x0099, 0x0190,
        0x0f00, 0x0e09, 0x0d03, 0x0c0a, 0x0b06, 0x0a0f, 0x0905, 0x080c,
        0x070c, 0x0605, 0x050f, 0x0406, 0x030a, 0x0203, 0x0109, 0x0000
    };

    extern const int kTriTable[256][16] = {
        {-1},
        {0, 8, 3, -1},
        {0, 1, 9, -1},
        {1, 8, 3, 9, 8, 1, -1},
        {1, 2, 10, -1},
        {0, 8, 3, 1, 2, 10, -1},
        {9, 2, 10, 0, 2, 9, -1},
        {2, 8, 3, 2, 10, 8, 10, 9, 8, -1},
        {3, 11, 2, -1},
        {0, 11, 2, 8, 11, 0, -1},
        {1, 9, 0, 2, 3, 11, -1},
        {1, 11, 2, 1, 9, 11, 9, 8, 11, -1},
        {3, 10, 1, 11, 10, 3, -1},
        {0, 10, 1, 0, 8, 10, 8, 11, 10, -1},
        {3, 9, 0, 3, 11, 9, 11, 10, 9, -1},
        {9, 8, 10, 10, 8, 11, -1},
        {4, 7, 8, -1},
        {4, 3, 0, 7, 3, 4, -1},
        {0, 1, 9, 8, 4, 7, -1},
        {4, 1, 9, 4, 7, 1, 7, 3, 1, -1},
        {1, 2, 10, 8, 4, 7, -1},
        {3, 4, 7, 3, 0, 4, 1, 2, 10, -1},
        {9, 2, 10, 9, 0, 2, 8, 4, 7, -1},
        {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1},
        {8, 4, 7, 3, 11, 2, -1},
        {11, 4, 7, 11, 2, 4, 2, 0, 4, -1},
        {9, 0, 1, 8, 4, 7, 2, 3, 11, -1},
        {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1},
        {3, 10, 1, 3, 11, 10, 7, 8, 4, -1},
        {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1},
        {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1},
        {4, 7, 11, 4, 11, 9, 9, 11, 10, -1},
        {9, 5, 4, -1},
        {9, 5, 4, 0, 8, 3, -1},
        {0, 5, 4, 1, 5, 0, -1},
        {8, 5, 4, 8, 3, 5, 3, 1, 5, -1},
        {1, 2, 10, 9, 5, 4, -1},
        {3, 0, 8, 1, 2, 10, 4, 9, 5, -1},
        {5, 2, 10, 5, 4, 2, 4, 0, 2, -1},
        {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1},
        {9, 5, 4, 2, 3, 11, -1},
        {0, 11, 2, 0, 8, 11, 4, 9, 5, -1},
        {0, 5, 4, 0, 1, 5, 2, 3, 11, -1},
        {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1},
        {10, 3, 11, 10, 1, 3, 9, 5, 4, -1},
        {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1},
        {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1},
        {5, 4, 8, 5, 8, 10, 10, 8, 11, -1},
        {9, 7, 8, 5, 7, 9, -1},
        {9, 3, 0, 9, 5, 3, 5, 7, 3, -1},
        {0, 7, 8, 0, 1, 7, 1, 5, 7, -1},
        {1, 5, 3, 3, 5, 7, -1},
        {9, 7, 8, 9, 5, 7, 10, 1, 2, -1},
        {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1},
        {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1},
        {2, 10, 5, 2, 5, 3, 3, 5, 7, -1},
        {7, 9, 5, 7, 8, 9, 3, 11, 2, -1},
        {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1},
        {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1},
        {11, 2, 1, 11, 1, 7, 7, 1, 5, -1},
        {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1},
        {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
        {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
        {11, 10, 5, 7, 11, 5, -1},
        {10, 6, 5, -1},
        {0, 8, 3, 5, 10, 6, -1},
        {9, 0, 1, 5, 10, 6, -1},
        {1, 8, 3, 1, 9, 8, 5, 10, 6, -1},
        {1, 6, 5, 2, 6, 1, -1},
        {1, 6, 5, 1, 2, 6, 3, 0, 8, -1},
        {9, 6, 5, 9, 0, 6, 0, 2, 6, -1},
        {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1},
        {2, 3, 11, 10, 6, 5, -1},
        {11, 0, 8, 11, 2, 0, 10, 6, 5, -1},
        {0, 1, 9, 2, 3, 11, 5, 10, 6, -1},
        {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1},
        {6, 3, 11, 6, 5, 3, 5, 1, 3, -1},
        {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1},
        {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1},
        {6, 5, 9, 6, 9, 11, 11, 9, 8, -1},
        {5, 10, 6, 4, 7, 8, -1},
        {4, 3, 0, 4, 7, 3, 6, 5, 10, -1},
        {1, 9, 0, 5, 10, 6, 8, 4, 7, -1},
        {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1},
        {6, 1, 2, 6, 5, 1, 4, 7, 8, -1},
        {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1},
        {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1},
        {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
        {3, 11, 2, 7, 8, 4, 10, 6, 5, -1},
        {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1},
        {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1},
        {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
        {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1},
        {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
        {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
        {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1},
        {10, 4, 9, 6, 4, 10, -1},
        {4, 10, 6, 4, 9, 10, 0, 8, 3, -1},
        {10, 0, 1, 10, 6, 0, 6, 4, 0, -1},
        {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1},
        {1, 4, 9, 1, 2, 4, 2, 6, 4, -1},
        {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1},
        {0, 2, 4, 4, 2, 6, -1},
        {8, 3, 2, 8, 2, 4, 4, 2, 6, -1},
        {10, 4, 9, 10, 6, 4, 11, 2, 3, -1},
        {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1},
        {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1},
        {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
        {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1},
        {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
        {3, 11, 6, 3, 6, 0, 0, 6, 4, -1},
        {6, 4, 8, 11, 6, 8, -1},
        {7, 10, 6, 7, 8, 10, 8, 9, 10, -1},
        {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1},
        {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1},
        {10, 6, 7, 10, 7, 1, 1, 7, 3, -1},
        {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1},
        {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
        {7, 8, 0, 7, 0, 6, 6, 0, 2, -1},
        {7, 3, 2, 6, 7, 2, -1},
        {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1},
        {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
        {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
        {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1},
        {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
        {0, 9, 1, 11, 6, 7, -1},
        {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1},
        {7, 11, 6, -1},
        {7, 6, 11, -1},
        {3, 0, 8, 11, 7, 6, -1},
        {0, 1, 9, 11, 7, 6, -1},
        {8, 1, 9, 8, 3, 1, 11, 7, 6, -1},
        {10, 1, 2, 6, 11, 7, -1},
        {1, 2, 10, 3, 0, 8, 6, 11, 7, -1},
        {2, 9, 0, 2, 10, 9, 6, 11, 7, -1},
        {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1},
        {7, 2, 3, 6, 2, 7, -1},
        {7, 0, 8, 7, 6, 0, 6, 2, 0, -1},
        {2, 7, 6, 2, 3, 7, 0, 1, 9, -1},
        {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1},
        {10, 7, 6, 10, 1, 7, 1, 3, 7, -1},
        {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1},
        {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1},
        {7, 6, 10, 7, 10, 8, 8, 10, 9, -1},
        {6, 8, 4, 11, 8, 6, -1},
        {3, 6, 11, 3, 0, 6, 0, 4, 6, -1},
        {8, 6, 11, 8, 4, 6, 9, 0, 1, -1},
        {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1},
        {6, 8, 4, 6, 11, 8, 2, 10, 1, -1},
        {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1},
        {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1},
        {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
        {8, 2, 3, 8, 4, 2, 4, 6, 2, -1},
        {0, 4, 2, 4, 6, 2, -1},
        {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1},
        {1, 9, 4, 1, 4, 2, 2, 4, 6, -1},
        {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1},
        {10, 1, 0, 10, 0, 6, 6, 0, 4, -1},
        {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
        {10, 9, 4, 6, 10, 4, -1},
        {4, 9, 5, 7, 6, 11, -1},
        {0, 8, 3, 4, 9, 5, 11, 7, 6, -1},
        {5, 0, 1, 5, 4, 0, 7, 6, 11, -1},
        {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1},
        {9, 5, 4, 10, 1, 2, 7, 6, 11, -1},
        {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1},
        {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1},
        {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
        {7, 2, 3, 7, 6, 2, 5, 4, 9, -1},
        {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1},
        {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1},
        {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
        {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1},
        {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
        {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
        {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1},
        {6, 9, 5, 6, 11, 9, 11, 8, 9, -1},
        {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1},
        {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1},
        {6, 11, 3, 6, 3, 5, 5, 3, 1, -1},
        {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1},
        {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
        {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
        {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1},
        {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1},
        {9, 5, 6, 9, 6, 0, 0, 6, 2, -1},
        {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
        {1, 5, 6, 2, 1, 6, -1},
        {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
        {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1},
        {0, 3, 8, 5, 6, 10, -1},
        {10, 5, 6, -1},
        {11, 5, 10, 7, 5, 11, -1},
        {11, 5, 10, 11, 7, 5, 8, 3, 0, -1},
        {5, 11, 7, 5, 10, 11, 1, 9, 0, -1},
        {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1},
        {11, 1, 2, 11, 7, 1, 7, 5, 1, -1},
        {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1},
        {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1},
        {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
        {2, 5, 10, 2, 3, 5, 3, 7, 5, -1},
        {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1},
        {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1},
        {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
        {1, 3, 5, 3, 7, 5, -1},
        {0, 8, 7, 0, 7, 1, 1, 7, 5, -1},
        {9, 0, 3, 9, 3, 5, 5, 3, 7, -1},
        {9, 8, 7, 5, 9, 7, -1},
        {5, 8, 4, 5, 10, 8, 10, 11, 8, -1},
        {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1},
        {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1},
        {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
        {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1},
        {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
        {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
        {9, 4, 5, 2, 11, 3, -1},
        {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1},
        {5, 10, 2, 5, 2, 4, 4, 2, 0, -1},
        {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
        {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1},
        {8, 4, 5, 8, 5, 3, 3, 5, 1, -1},
        {0, 4, 5, 1, 0, 5, -1},
        {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1},
        {9, 4, 5, -1},
        {4, 11, 7, 4, 9, 11, 9, 10, 11, -1},
        {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1},
        {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1},
        {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
        {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1},
        {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
        {11, 7, 4, 11, 4, 2, 2, 4, 0, -1},
        {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1},
        {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1},
        {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
        {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
        {1, 10, 2, 8, 7, 4, -1},
        {4, 9, 1, 4, 1, 7, 7, 1, 3, -1},
        {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1},
        {4, 0, 3, 7, 4, 3, -1},
        {4, 8, 7, -1},
        {9, 10, 8, 10, 11, 8, -1},
        {3, 0, 9, 3, 9, 11, 11, 9, 10, -1},
        {0, 1, 10, 0, 10, 8, 8, 10, 11, -1},
        {3, 1, 10, 11, 3, 10, -1},
        {1, 2, 11, 1, 11, 9, 9, 11, 8, -1},
        {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1},
        {0, 2, 11, 8, 0, 11, -1},
        {3, 2, 11, -1},
        {2, 3, 8, 2, 8, 10, 10, 8, 9, -1},
        {9, 10, 2, 0, 9, 2, -1},
        {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1},
        {1, 10, 2, -1},
        {1, 3, 8, 9, 1, 8, -1},
        {0, 9, 1, -1},
        {0, 3, 8, -1},
        {-1}
    };


    int extractSurfaceTriangles(
        const SPHParticles::Data* d_particles,
        uint32_t num_particles,
        const AABB& wall,
        int grid_resolution,
        float iso_level,
        float support_radius_scale,
        int field_smooth_iters,
        float field_smooth_weight,
        std::vector<Vec3f>& out_triangle_vertices,
        int max_triangles = std::numeric_limits<int>::max()
    )
    {
        if (!d_particles || num_particles == 0) {
            out_triangle_vertices.clear();
            return 0;
        }

        const int n = std::max(8, grid_resolution);
        const int nx = n;
        const int ny = n;
        const int nz = n;

        std::vector<float> field;
        float* d_field = nullptr;
        buildSurfaceFieldCUDA(
            d_particles,
            num_particles,
            wall,
            nx,
            ny,
            nz,
            support_radius_scale,
            field_smooth_iters,
            field_smooth_weight,
            field,
            &d_field,
            0
        );
        extractSurfaceTrianglesCUDA(
            d_field,
            nx,
            ny,
            nz,
            wall,
            iso_level,
            out_triangle_vertices,
            max_triangles,
            0
        );

        CUDA_CHECK(cudaFree(d_field));
        return static_cast<int>(out_triangle_vertices.size() / 3);
    }
}

void App::exportSurfaceOBJ()
{
    const uint32_t num_particles = particles->numPrimitives();
    if (num_particles == 0) {
        printf("[SurfaceExport] No particles to export.\n");
        return;
    }

    std::vector<Vec3f> tri_vertices;
    const int tri_count = marching_cubes_tables::extractSurfaceTriangles(
        reinterpret_cast<const SPHParticles::Data*>(particles->devicePtr()),
        num_particles,
        wall,
        surface_grid_resolution,
        surface_iso_level,
        surface_support_radius_scale,
        surface_field_smooth_iters,
        surface_field_smooth_weight,
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
    const int max_tris = marching_cubes_tables::maxTrianglesForGrid(n);
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
    auto& surface_faces = const_cast<std::vector<Face>&>(surface_mesh->faces());
    for (auto& face : surface_faces) {
        face.texcoord_id = face.vertex_id;
        face.normal_id = face.vertex_id;
    }
    auto& surface_texcoords = const_cast<std::vector<Vec2f>&>(surface_mesh->texcoords());
    surface_texcoords.assign(surface_mesh->vertices().size(), Vec2f(0.0f));
    std::vector<Vec3f> surface_normals;
    computeTriangleMeshSmoothNormalsCUDA(surface_mesh->vertices(), surface_mesh->faces(), surface_normals, stream);
    surface_mesh->setNormals(surface_normals);
    surface_mesh->copyToDevice();
}

void App::updateRealtimeSurfaceMesh()
{
    if (!surface_mesh)
        return;

    const uint32_t num_particles = particles->numPrimitives();
    if (num_particles == 0)
        return;

    std::vector<Vec3f> tri_vertices;
    const int max_tris = (int)surface_runtime_faces.size();
    const int tri_count = marching_cubes_tables::extractSurfaceTriangles(
        reinterpret_cast<const SPHParticles::Data*>(particles->devicePtr()),
        num_particles,
        wall,
        surface_runtime_resolution,
        surface_iso_level,
        surface_support_radius_scale,
        surface_field_smooth_iters,
        surface_field_smooth_weight,
        tri_vertices,
        max_tris
    );

    const Vec3f hidden_pos(0.0f, surface_hidden_offset_y, 0.0f);
    std::fill(surface_runtime_vertices.begin(), surface_runtime_vertices.end(), hidden_pos);

    // Weld near-identical marching-cubes vertices so adjacent triangles actually
    // share vertex IDs; otherwise normal averaging degenerates to per-face normals.
    struct QuantizedPos {
        int x;
        int y;
        int z;
        bool operator==(const QuantizedPos& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    struct QuantizedPosHash {
        size_t operator()(const QuantizedPos& q) const
        {
            const size_t h1 = std::hash<int>{}(q.x);
            const size_t h2 = std::hash<int>{}(q.y);
            const size_t h3 = std::hash<int>{}(q.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    const float wall_extent_x = wall.max().x() - wall.min().x();
    const float weld_epsilon = std::max(1e-5f, wall_extent_x / std::max(8, surface_runtime_resolution) * 1e-3f);
    auto quantize = [weld_epsilon](const Vec3f& p) -> QuantizedPos {
        return QuantizedPos{
            static_cast<int>(std::floor(p.x() / weld_epsilon + 0.5f)),
            static_cast<int>(std::floor(p.y() / weld_epsilon + 0.5f)),
            static_cast<int>(std::floor(p.z() / weld_epsilon + 0.5f))
        };
    };

    std::unordered_map<QuantizedPos, int, QuantizedPosHash> welded_ids;
    welded_ids.reserve(tri_vertices.size());

    const int hidden_idx = static_cast<int>(surface_runtime_vertices.size()) - 1;
    int next_vertex_idx = 0;
    for (int t = 0; t < max_tris; ++t) {
        if (t < tri_count) {
            Vec3i ids(0);
            for (int k = 0; k < 3; ++k) {
                const Vec3f& p = tri_vertices[(size_t)t * 3 + (size_t)k];
                const QuantizedPos q = quantize(p);

                auto it = welded_ids.find(q);
                if (it == welded_ids.end()) {
                    int id = hidden_idx;
                    if (next_vertex_idx < hidden_idx) {
                        id = next_vertex_idx;
                        surface_runtime_vertices[(size_t)id] = p;
                        ++next_vertex_idx;
                    }
                    welded_ids.emplace(q, id);
                    ids[k] = id;
                }
                else {
                    ids[k] = it->second;
                }
            }

            Face& face = surface_runtime_faces[(size_t)t];
            face.vertex_id = ids;
            face.normal_id = ids;
            face.texcoord_id = ids;
        }
        else {
            Face& face = surface_runtime_faces[(size_t)t];
            const Vec3i hidden_face(hidden_idx);
            face.vertex_id = hidden_face;
            face.normal_id = hidden_face;
            face.texcoord_id = hidden_face;
        }
    }

    auto& vertices = const_cast<std::vector<Vec3f>&>(surface_mesh->vertices());
    auto& faces = const_cast<std::vector<Face>&>(surface_mesh->faces());
    if (vertices.size() != surface_runtime_vertices.size())
        return;
    if (faces.size() != surface_runtime_faces.size())
        return;

    std::copy(surface_runtime_vertices.begin(), surface_runtime_vertices.end(), vertices.begin());
    std::copy(surface_runtime_faces.begin(), surface_runtime_faces.end(), faces.begin());
    std::vector<Vec3f> surface_normals;
    computeTriangleMeshSmoothNormalsCUDA(surface_mesh->vertices(), surface_mesh->faces(), surface_normals, stream);
    surface_mesh->setNormals(surface_normals);
    surface_mesh->copyToDevice();

    scene.updateObjectGAS("fluidSurface", context, stream);

    if ((params.frame % 30) == 0) {
        printf("[SurfaceMode] updated mesh triangles=%d/%d\n", tri_count, max_tris);
    }
}

void App::buildSceneForMode(FluidRenderMode mode)
{
    // Clear all scene objects so the SBT can be rebuilt from a clean cursor.
    scene.clearObjects();
    printf("[FluidMode] rebuild scene mode=%s\n", mode == FluidRenderMode::Particles ? "Particles" : "Surface");

    // Re-add objects in a canonical order.
    scene.addObject("floor", make_shared<Plane>(Vec2f(-75.0f, -75.0f), Vec2f(75.0f, 75.0f)), floor_bsdf, { floor_hitgroup_prg }, Matrix4f::translate(0, -100.0f, 0));

    if (mode == FluidRenderMode::Particles) {
        scene.addObject("particles", particles, particle_bsdf, { particle_hitgroup_prg }, Matrix4f::identity(), {true, true});
    }
    else if (mode == FluidRenderMode::Surface) {
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
        .time_step = 0.1f,
        .stiffness = 8.0f,
        .viscosity = 0.02f,
        .ks = 25.0f,
        .kd = 20.0f
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
    params.samples_per_launch = 32;
    params.frame = 0;
    params.max_depth = 15;
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
    particle_bsdf = make_shared<Dielectric>(dielectric_id, make_shared<ConstantTexture>(Vec3f(0.75f), constant_id), 1.5f);
    surface_bsdf = make_shared<Dielectric>(dielectric_id, make_shared<ConstantTexture>(Vec3f(0.75f), constant_id), 1.5f);

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

    buildSceneForMode(render_mode);
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
        buildSceneForMode(render_mode);
    }

    if (render_mode == FluidRenderMode::Particles) {
        scene.updateObjectGAS("particles", context, stream);
    }
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
        ImGui::SliderInt("Surface Runtime Resolution", &surface_runtime_resolution, 8, 48);
        ImGui::SliderFloat("Surface Hidden Offset Y", &surface_hidden_offset_y, -200000.0f, -10000.0f);
        ImGui::SliderInt("Surface Field Smooth Iters", &surface_field_smooth_iters, 0, 6);
        ImGui::SliderFloat("Surface Field Smooth Weight", &surface_field_smooth_weight, 0.0f, 1.0f);
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



