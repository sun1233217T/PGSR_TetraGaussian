#include "rasterizer_torch.h"

#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <torch/extension.h>
#include <torch/torch.h>

#include "voxel_initialize.h"

using torch::indexing::Slice;

namespace {

std::pair<torch::Tensor, torch::Tensor> build_rays_from_camera(
    const torch::Tensor& intrinsic,
    const torch::Tensor& extrinsic,
    int64_t height,
    int64_t width,
    torch::Device device) {
    if (height <= 0 || width <= 0) throw std::invalid_argument("height/width must be positive");
    // 统一为 float32，在指定 device 上构建。
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto K = intrinsic.to(opts);
    auto ext = extrinsic.to(opts);

    auto cam2world = torch::linalg::inv(ext);
    auto R = cam2world.index({Slice(0, 3), Slice(0, 3)});           // (3,3)
    auto t = cam2world.index({Slice(0, 3), 3}).reshape({1, 3});     // (1,3)

    // 像素中心坐标
    auto xs = torch::arange(width, opts).add(0.5f);
    auto ys = torch::arange(height, opts).add(0.5f);
    auto grids = torch::meshgrid({ys, xs}, "ij");
    auto ys_grid = grids[0];
    auto xs_grid = grids[1];

    const float fx = K.index({0, 0}).item<float>();
    const float fy = K.index({1, 1}).item<float>();
    const float cx = K.index({0, 2}).item<float>();
    const float cy = K.index({1, 2}).item<float>();

    auto dirs_cam = torch::stack(
        std::initializer_list<torch::Tensor>{
            (xs_grid - cx) / fx,
            (ys_grid - cy) / fy,
            torch::ones_like(xs_grid)
        }, -1); // (H,W,3)
    dirs_cam = dirs_cam.reshape({-1, 3});
    dirs_cam = dirs_cam / torch::norm(dirs_cam, 2, -1, true);
    auto dirs_world = torch::matmul(dirs_cam, R.transpose(0, 1)); // (N,3)
    auto origins = t.expand_as(dirs_world);                       // (N,3)
    return {origins, dirs_world};
}

void check_matrix(const torch::Tensor& t, int64_t rows, int64_t cols, const char* name) {
    if (t.dim() != 2 || t.size(0) != rows || t.size(1) != cols) {
        throw std::invalid_argument(std::string(name) + " must have shape (" +
            std::to_string(rows) + "," + std::to_string(cols) + ")");
    }
}

} // namespace

torch::Tensor rasterize_image(
    VoxelGrid& grid,
    const torch::Tensor& intrinsic,
    const torch::Tensor& extrinsic,
    int64_t height,
    int64_t width,
    int64_t coarse_res) {
    check_matrix(intrinsic, 3, 3, "intrinsic");
    check_matrix(extrinsic, 4, 4, "extrinsic");
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is required for rasterize_image");
    }
    torch::Device device(torch::kCUDA);

    // 构建 coarse 索引，直接放在 CUDA 上
    auto coarse_tuple = build_coarse_index(grid, coarse_res, /*on_cuda=*/true);
    torch::Tensor occ = std::get<0>(coarse_tuple);
    torch::Tensor offsets = std::get<1>(coarse_tuple);
    torch::Tensor keys = std::get<2>(coarse_tuple);
    torch::Tensor mask = std::get<3>(coarse_tuple);
    (void)occ; // 目前未在占位渲染中使用

    // 相机参数 -> 射线
    auto rays = build_rays_from_camera(intrinsic, extrinsic, height, width, device);
    torch::Tensor rays_o = rays.first.contiguous();
    torch::Tensor rays_d = rays.second.contiguous();

    Vec3 origin = grid.origin();
    Vec3i dims = grid.dims();
    double voxel_size = grid.voxel_size();
    if (dims.x <= 0 || dims.y <= 0 || dims.z <= 0) {
        throw std::runtime_error("grid dims are zero; cannot rasterize");
    }
    int64_t brick_x = static_cast<int64_t>(std::ceil(static_cast<double>(dims.x) / coarse_res));
    int64_t brick_y = static_cast<int64_t>(std::ceil(static_cast<double>(dims.y) / coarse_res));
    int64_t brick_z = static_cast<int64_t>(std::ceil(static_cast<double>(dims.z) / coarse_res));

    return rasterize_forward_cuda(
        rays_o, rays_d, offsets, keys, mask,
        height, width,
        static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(origin.z),
        static_cast<float>(voxel_size),
        dims.x, dims.y, dims.z,
        brick_x, brick_y, brick_z,
        coarse_res);
}

torch::Tensor rasterize_image_with_index(
    VoxelGrid& grid,
    const torch::Tensor& intrinsic,
    const torch::Tensor& extrinsic,
    const torch::Tensor& coarse_offsets,
    const torch::Tensor& voxel_keys,
    const torch::Tensor& coarse_mask,
    int64_t height,
    int64_t width,
    int64_t coarse_res) {
    check_matrix(intrinsic, 3, 3, "intrinsic");
    check_matrix(extrinsic, 4, 4, "extrinsic");
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is required for rasterize_image_with_index");
    }
    torch::Device device(torch::kCUDA);

    // 相机参数 -> 射线
    auto rays = build_rays_from_camera(intrinsic, extrinsic, height, width, device);
    torch::Tensor rays_o = rays.first.contiguous();
    torch::Tensor rays_d = rays.second.contiguous();

    Vec3 origin = grid.origin();
    Vec3i dims = grid.dims();
    double voxel_size = grid.voxel_size();
    if (dims.x <= 0 || dims.y <= 0 || dims.z <= 0) {
        throw std::runtime_error("grid dims are zero; cannot rasterize");
    }
    int64_t brick_x = static_cast<int64_t>(std::ceil(static_cast<double>(dims.x) / coarse_res));
    int64_t brick_y = static_cast<int64_t>(std::ceil(static_cast<double>(dims.y) / coarse_res));
    int64_t brick_z = static_cast<int64_t>(std::ceil(static_cast<double>(dims.z) / coarse_res));

    // 将预计算的 coarse 索引搬到 GPU
    auto offsets = coarse_offsets.to(device).contiguous();
    auto keys = voxel_keys.to(device).contiguous();
    auto mask = coarse_mask.to(device).contiguous();

    return rasterize_forward_cuda(
        rays_o, rays_d, offsets, keys, mask,
        height, width,
        static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(origin.z),
        static_cast<float>(voxel_size),
        dims.x, dims.y, dims.z,
        brick_x, brick_y, brick_z,
        coarse_res);
}
