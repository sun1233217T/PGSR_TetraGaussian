#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "voxel_grid.h"

// 将所有顶点属性初始化为 IntAttribute(1)
void grid_init(VoxelGrid& grid);

// 输入射线 origins (N,3) 与 dirs (N,M,3) 或 (K,3)，可达体素顶点属性置 0
void image_pre_resterization(VoxelGrid& grid, const pybind11::array& origins, const pybind11::array& dirs);

// 将 dense 网格稀疏化：仅保留顶点属性为 1 的体素，顶点属性清空。
// tolerant=0 仅判断自身；1 为 8 邻域 (同层 3x3)，2 为 26 邻域 (3x3x3) 容错。
// invert=true 时反选（原本保留的剔除，原本剔除的保留）。
VoxelGrid grad_sparsilization(const VoxelGrid& dense, int tolerant = 0, bool invert = false);
