#include "voxel_utility.h"

#include <algorithm>

std::vector<std::pair<VoxelKey, std::shared_ptr<Vertex>>> sorted_vertices(VoxelGrid& grid) {
    std::vector<std::pair<VoxelKey, std::shared_ptr<Vertex>>> items;
    items.reserve(grid.vertex_count());
    for (const auto& kv : grid.vertices()) {
        items.emplace_back(kv.first, kv.second);
    }
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.first.x != b.first.x) return a.first.x < b.first.x;
        if (a.first.y != b.first.y) return a.first.y < b.first.y;
        return a.first.z < b.first.z;
    });
    return items;
}
