#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "voxel_grid.h"

// 辅助工具：收集按 key 稳定排序的顶点列表，便于构造确定性的张量行顺序。
std::vector<std::pair<VoxelKey, std::shared_ptr<Vertex>>> sorted_vertices(VoxelGrid& grid);

// 检查体素索引是否处于网格维度范围内。
inline bool cell_index_inside(const Vec3i& idx, const Vec3i& dims) {
    return idx.x >= 0 && idx.y >= 0 && idx.z >= 0 &&
           idx.x < dims.x && idx.y < dims.y && idx.z < dims.z;
}
