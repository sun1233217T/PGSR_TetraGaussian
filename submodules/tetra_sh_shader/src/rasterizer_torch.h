#pragma once

#include <torch/extension.h>

#include "voxel_grid.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> build_dense_vertex_grids_from_features(
    VoxelGrid& grid,
    const torch::Tensor& vertex_features);

// 高层接口：输入 VoxelGrid 和相机参数，输出 HxWx3 颜色（占位）。
torch::Tensor rasterize_image(
    VoxelGrid& grid,
    const torch::Tensor& intrinsic,   // (3,3) float/double
    const torch::Tensor& extrinsic,   // (4,4) world->cam, float/double
    int64_t height,
    int64_t width,
    int64_t coarse_res = 8,
    const torch::Tensor& vertex_features = torch::Tensor());

// 使用预计算的 coarse 索引进行渲染（避免每帧重建 coarse 索引）
torch::Tensor rasterize_image_with_index(
    VoxelGrid& grid,
    const torch::Tensor& intrinsic,
    const torch::Tensor& extrinsic,
    const torch::Tensor& coarse_offsets,
    const torch::Tensor& voxel_keys,
    const torch::Tensor& coarse_mask,
    int64_t height,
    int64_t width,
    int64_t coarse_res = 8,
    const torch::Tensor& vertex_features = torch::Tensor());

// 使用预计算 coarse 索引 + 预计算稠密顶点网格（避免重复构建）
torch::Tensor rasterize_image_with_index_dense(
    VoxelGrid& grid,
    const torch::Tensor& intrinsic,
    const torch::Tensor& extrinsic,
    const torch::Tensor& coarse_offsets,
    const torch::Tensor& voxel_keys,
    const torch::Tensor& coarse_mask,
    const torch::Tensor& vertex_sigma,
    const torch::Tensor& vertex_color,
    const torch::Tensor& vertex_mask,
    int64_t height,
    int64_t width,
    int64_t coarse_res = 8);

// CUDA 实现（占位）：rays_o/rays_d 为 (N,3)，offsets/keys/mask 为 coarse 索引。
torch::Tensor rasterize_forward_cuda(
    const torch::Tensor& rays_o,
    const torch::Tensor& rays_d,
    const torch::Tensor& coarse_offsets,
    const torch::Tensor& voxel_keys,
    const torch::Tensor& coarse_mask,
    const torch::Tensor& vertex_sigma,
    const torch::Tensor& vertex_color,
    const torch::Tensor& vertex_mask,
    int64_t height,
    int64_t width,
    float origin_x,
    float origin_y,
    float origin_z,
    float voxel_size,
    int64_t dim_x,
    int64_t dim_y,
    int64_t dim_z,
    int64_t brick_x,
    int64_t brick_y,
    int64_t brick_z,
    int64_t coarse_res);
