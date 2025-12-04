#pragma once

#include <torch/extension.h>

#include "voxel_grid.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> build_dense_vertex_grids_from_features(
    VoxelGrid& grid,
    const torch::Tensor& vertex_features);

// 按 coarse 砖打包的稠密顶点网格，避免为全局网格分配 (Dx+1)*(Dy+1)*(Dz+1)。
// 返回 (sigma_packed, color_packed, valid_packed, brick_starts, brick_shapes, brick_vertex_offsets)。
// brick_starts/shapes 为 int64[B,3]，vertex_offsets 为 int64[B+1] 前缀和。
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_packed_dense_vertex_grids_from_features(
    VoxelGrid& grid,
    const torch::Tensor& vertex_features,
    const torch::Tensor& coarse_offsets,
    int64_t coarse_res = 8,
    bool on_cuda = false);

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

// 使用按砖打包的稠密顶点网格进行渲染（节省全局稠密分配）。
torch::Tensor rasterize_image_with_index_packed(
    VoxelGrid& grid,
    const torch::Tensor& intrinsic,
    const torch::Tensor& extrinsic,
    const torch::Tensor& coarse_offsets,
    const torch::Tensor& voxel_keys,
    const torch::Tensor& coarse_mask,
    const torch::Tensor& vertex_sigma_packed,
    const torch::Tensor& vertex_color_packed,
    const torch::Tensor& vertex_mask_packed,
    const torch::Tensor& brick_starts,
    const torch::Tensor& brick_shapes,
    const torch::Tensor& brick_vertex_offsets,
    int64_t height,
    int64_t width,
    int64_t coarse_res = 8);

// 反向：仅对 dense 顶点 sigma/color 求梯度。
std::vector<torch::Tensor> rasterize_image_with_index_dense_backward(
    VoxelGrid& grid,
    const torch::Tensor& intrinsic,
    const torch::Tensor& extrinsic,
    const torch::Tensor& coarse_offsets,
    const torch::Tensor& voxel_keys,
    const torch::Tensor& coarse_mask,
    const torch::Tensor& vertex_sigma,
    const torch::Tensor& vertex_color,
    const torch::Tensor& vertex_mask,
    const torch::Tensor& grad_output,
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

// 反向：打包稠密网格的版本。
std::vector<torch::Tensor> rasterize_image_with_index_packed_backward(
    VoxelGrid& grid,
    const torch::Tensor& intrinsic,
    const torch::Tensor& extrinsic,
    const torch::Tensor& coarse_offsets,
    const torch::Tensor& voxel_keys,
    const torch::Tensor& coarse_mask,
    const torch::Tensor& vertex_sigma_packed,
    const torch::Tensor& vertex_color_packed,
    const torch::Tensor& vertex_mask_packed,
    const torch::Tensor& brick_starts,
    const torch::Tensor& brick_shapes,
    const torch::Tensor& brick_vertex_offsets,
    const torch::Tensor& grad_output,
    int64_t height,
    int64_t width,
    int64_t coarse_res);

// CUDA 前向（打包）。
torch::Tensor rasterize_forward_packed_cuda(
    const torch::Tensor& rays_o,
    const torch::Tensor& rays_d,
    const torch::Tensor& coarse_offsets,
    const torch::Tensor& voxel_keys,
    const torch::Tensor& coarse_mask,
    const torch::Tensor& vertex_sigma,
    const torch::Tensor& vertex_color,
    const torch::Tensor& vertex_mask,
    const torch::Tensor& brick_starts,
    const torch::Tensor& brick_shapes,
    const torch::Tensor& brick_vertex_offsets,
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

// 反向：仅对 dense 顶点 sigma / color 求梯度，不更新 coarse/射线等。
std::vector<torch::Tensor> rasterize_backward_cuda(
    const torch::Tensor& rays_o,
    const torch::Tensor& rays_d,
    const torch::Tensor& coarse_offsets,
    const torch::Tensor& voxel_keys,
    const torch::Tensor& coarse_mask,
    const torch::Tensor& vertex_sigma,
    const torch::Tensor& vertex_color,
    const torch::Tensor& vertex_mask,
    const torch::Tensor& grad_output,  // (H,W,3)
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

// 反向（打包）。
std::vector<torch::Tensor> rasterize_backward_packed_cuda(
    const torch::Tensor& rays_o,
    const torch::Tensor& rays_d,
    const torch::Tensor& coarse_offsets,
    const torch::Tensor& voxel_keys,
    const torch::Tensor& coarse_mask,
    const torch::Tensor& vertex_sigma,
    const torch::Tensor& vertex_color,
    const torch::Tensor& vertex_mask,
    const torch::Tensor& brick_starts,
    const torch::Tensor& brick_shapes,
    const torch::Tensor& brick_vertex_offsets,
    const torch::Tensor& grad_output,  // (H,W,3)
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
