#pragma once

#include <cstddef>
#include <torch/extension.h>

#include "voxel_grid.h"

// 分配/挂接一个 (N, C) torch 张量并把每个顶点的 attr 指向其对应行。
// 行顺序为按 VoxelKey (x,y,z) 排序后的顶点顺序，保持确定性。
torch::Tensor init_vertex_tensor(VoxelGrid& grid,
                                 int64_t feature_dim,
                                 bool on_cuda = false,
                                 bool fill_ones = false,
                                 bool as_int = false);

// 将已有的 (N, C) 张量挂接到顶点 attr；若非 contiguous 则会复制为 contiguous。
// 要求张量第一维与顶点数量一致。
torch::Tensor attach_vertex_tensor(VoxelGrid& grid, const torch::Tensor& tensor);

// 依据顶点是否有 attr，补齐所有“活跃”体素单元（八个角有任意一个带 attr 即视为活跃）。
// 返回新增的体素数量。
size_t ensure_active_cells_from_vertex_attr(VoxelGrid& grid);

// 构建粗粒度占据网格（默认 8x8x8）。返回 torch::Tensor (uint8)。
torch::Tensor build_coarse_occupancy(const VoxelGrid& grid, int64_t res = 8, bool on_cuda = false);

// 构建 coarse 索引：占据网格 + 砖偏移 + 体素列表 + 砖内粗 bitmask。
// 返回 (occupancy uint8[res,res,res], offsets int64[B+1], keys int64[N,3], bitmask uint64[B])
// 其中 B=res^3，offsets 给出每个砖的体素列表范围，keys 为按砖分桶压紧的 VoxelKey。
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_coarse_index(const VoxelGrid& grid, int64_t res = 8, bool on_cuda = false);
