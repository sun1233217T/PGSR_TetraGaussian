#include "rasterizer_torch.h"

#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <unordered_map>

#include <torch/extension.h>
#include <torch/torch.h>

#include "voxel_initialize.h"
#include "voxel_utility.h"

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> build_dense_vertex_grids_from_features(
    VoxelGrid& grid,
    const torch::Tensor& vertex_features) {
    if (!vertex_features.defined() || vertex_features.numel() == 0) {
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
    if (vertex_features.dim() != 2 || vertex_features.size(1) < 4) {
        throw std::invalid_argument("vertex_features must have shape (N,4) or (N,>=4)");
    }

    // key -> row index (sorted_vertices 顺序与 init_vertex_tensor 一致)
    auto ordered = sorted_vertices(grid);
    std::unordered_map<VoxelKey, int64_t, VoxelKeyHash> key_to_row;
    key_to_row.reserve(ordered.size());
    int64_t row = 0;
    for (const auto& kv : ordered) {
        key_to_row.emplace(kv.first, row++);
    }

    auto feats_cpu = vertex_features.to(torch::kCPU).contiguous().to(torch::kFloat32);

    Vec3i dims = grid.dims();
    torch::TensorOptions fopt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::TensorOptions bopt = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto sigma_cpu = torch::zeros({dims.x + 1, dims.y + 1, dims.z + 1}, fopt);
    auto color_cpu = torch::zeros({dims.x + 1, dims.y + 1, dims.z + 1, 3}, fopt);
    auto valid_cpu = torch::zeros({dims.x + 1, dims.y + 1, dims.z + 1}, bopt);
    auto xyz_cpu = torch::zeros({dims.x + 1, dims.y + 1, dims.z + 1, 3}, fopt);

    auto feats_acc = feats_cpu.accessor<float, 2>();
    auto sig_acc = sigma_cpu.accessor<float, 3>();
    auto col_acc = color_cpu.accessor<float, 4>();
    auto val_acc = valid_cpu.accessor<uint8_t, 3>();
    auto pos_acc = xyz_cpu.accessor<float, 4>();

    Vec3 origin = grid.origin();
    double vx = grid.voxel_size();
    for (int64_t x = 0; x <= dims.x; ++x) {
        for (int64_t y = 0; y <= dims.y; ++y) {
            for (int64_t z = 0; z <= dims.z; ++z) {
                pos_acc[x][y][z][0] = static_cast<float>(origin.x + static_cast<double>(x) * vx);
                pos_acc[x][y][z][1] = static_cast<float>(origin.y + static_cast<double>(y) * vx);
                pos_acc[x][y][z][2] = static_cast<float>(origin.z + static_cast<double>(z) * vx);
            }
        }
    }
    for (const auto& kv : grid.vertices()) {
        auto it = key_to_row.find(kv.first);
        if (it == key_to_row.end()) continue;
        const int64_t r = it->second;
        const int64_t x = kv.first.x;
        const int64_t y = kv.first.y;
        const int64_t z = kv.first.z;
        if (x < 0 || y < 0 || z < 0 || x > dims.x || y > dims.y || z > dims.z) continue;
        sig_acc[x][y][z] = feats_acc[r][0];
        col_acc[x][y][z][0] = feats_acc[r][1];
        col_acc[x][y][z][1] = feats_acc[r][2];
        col_acc[x][y][z][2] = feats_acc[r][3];
        val_acc[x][y][z] = 1;
    }
    return {sigma_cpu, color_cpu, valid_cpu, xyz_cpu};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_packed_dense_vertex_grids_from_features(
    VoxelGrid& grid,
    const torch::Tensor& vertex_features,
    const torch::Tensor& coarse_offsets,
    int64_t coarse_res,
    bool on_cuda) {
    if (!vertex_features.defined() || vertex_features.numel() == 0) {
        auto opts_i = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor empty_i = torch::zeros({0}, opts_i);
        torch::Tensor empty_f = torch::zeros({0}, opts_f);
        torch::Tensor empty_u8 = torch::zeros({0}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        return {empty_f, torch::zeros({0, 3}, opts_f), empty_u8, empty_i.reshape({0, 3}), empty_i.reshape({0, 3}), empty_i};
    }
    if (vertex_features.dim() != 2 || vertex_features.size(1) < 4) {
        throw std::invalid_argument("vertex_features must have shape (N,4) or (N,>=4)");
    }

    const int64_t B = coarse_res * coarse_res * coarse_res;
    auto offsets_cpu = coarse_offsets.to(torch::kCPU).contiguous();
    auto offsets_acc = offsets_cpu.accessor<int64_t, 1>();

    Vec3i dims = grid.dims();
    if (dims.x <= 0 || dims.y <= 0 || dims.z <= 0) {
        throw std::invalid_argument("grid dims must be positive for packing");
    }
    const int64_t brick_size_x = static_cast<int64_t>(std::ceil(static_cast<double>(dims.x) / coarse_res));
    const int64_t brick_size_y = static_cast<int64_t>(std::ceil(static_cast<double>(dims.y) / coarse_res));
    const int64_t brick_size_z = static_cast<int64_t>(std::ceil(static_cast<double>(dims.z) / coarse_res));

    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto brick_starts = torch::zeros({B, 3}, opts_i64);
    auto brick_shapes = torch::zeros({B, 3}, opts_i64);
    auto brick_vertex_offsets = torch::zeros({B + 1}, opts_i64);

    auto starts_acc = brick_starts.accessor<int64_t, 2>();
    auto shapes_acc = brick_shapes.accessor<int64_t, 2>();
    auto vert_off_acc = brick_vertex_offsets.accessor<int64_t, 1>();

    // prefix sum of vertex counts per brick
    for (int64_t b = 0; b < B; ++b) {
        const int64_t off0 = offsets_acc[b];
        const int64_t off1 = offsets_acc[b + 1];
        if (off1 == off0) {
            vert_off_acc[b + 1] = vert_off_acc[b];
            continue;
        }
        const int64_t bx = b / (coarse_res * coarse_res);
        const int64_t by = (b / coarse_res) % coarse_res;
        const int64_t bz = b % coarse_res;
        const int64_t start_x = bx * brick_size_x;
        const int64_t start_y = by * brick_size_y;
        const int64_t start_z = bz * brick_size_z;
        const int64_t size_x = std::min<int64_t>(brick_size_x, dims.x - start_x);
        const int64_t size_y = std::min<int64_t>(brick_size_y, dims.y - start_y);
        const int64_t size_z = std::min<int64_t>(brick_size_z, dims.z - start_z);
        starts_acc[b][0] = start_x;
        starts_acc[b][1] = start_y;
        starts_acc[b][2] = start_z;
        shapes_acc[b][0] = size_x;
        shapes_acc[b][1] = size_y;
        shapes_acc[b][2] = size_z;
        const int64_t verts = (size_x + 1) * (size_y + 1) * (size_z + 1);
        vert_off_acc[b + 1] = vert_off_acc[b] + verts;
    }
    const int64_t total_verts = vert_off_acc[B];

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto sigma_cpu = torch::zeros({total_verts}, opts_f32);
    auto color_cpu = torch::zeros({total_verts, 3}, opts_f32);
    auto valid_cpu = torch::zeros({total_verts}, opts_u8);

    // key -> row idx
    auto ordered = sorted_vertices(grid);
    std::unordered_map<VoxelKey, int64_t, VoxelKeyHash> key_to_row;
    key_to_row.reserve(ordered.size());
    int64_t row = 0;
    for (const auto& kv : ordered) {
        key_to_row.emplace(kv.first, row++);
    }
    auto feats_cpu = vertex_features.to(torch::kCPU).contiguous().to(torch::kFloat32);
    auto feats_acc = feats_cpu.accessor<float, 2>();
    auto starts_a = brick_starts.accessor<int64_t, 2>();
    auto shapes_a = brick_shapes.accessor<int64_t, 2>();
    auto sig_ptr = sigma_cpu.data_ptr<float>();
    auto col_ptr = color_cpu.data_ptr<float>();
    auto val_ptr = valid_cpu.data_ptr<uint8_t>();

    // helper for vertex index in packed buffer
    auto vert_index = [&](int64_t b, int64_t lx, int64_t ly, int64_t lz) -> int64_t {
        const int64_t sx = shapes_a[b][0] + 1;
        const int64_t sy = shapes_a[b][1] + 1;
        const int64_t sz = shapes_a[b][2] + 1;
        const int64_t base = vert_off_acc[b];
        return base + (lx * sy + ly) * sz + lz;
    };

    // voxel keys tensor is torch::Tensor? coarse_offsets only passed; reuse sorted grid.cells? Instead rely on offsets+grid.cells order can't.
    // We still need voxel keys list; rebuild from grid.cells sorted by brick for determinism.
    // Collect voxel keys by brick to avoid passing another tensor.
    std::vector<std::vector<VoxelKey>> brick_keys(B);
    for (const auto& kv : grid.cells()) {
        const VoxelKey& ck = kv.first;
        int64_t bx = ck.x / brick_size_x;
        int64_t by = ck.y / brick_size_y;
        int64_t bz = ck.z / brick_size_z;
        if (bx >= coarse_res) bx = coarse_res - 1;
        if (by >= coarse_res) by = coarse_res - 1;
        if (bz >= coarse_res) bz = coarse_res - 1;
        int64_t b = (bx * coarse_res + by) * coarse_res + bz;
        brick_keys[b].push_back(ck);
    }

    const int dx[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    const int dy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    const int dz[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    for (int64_t b = 0; b < B; ++b) {
        if (shapes_a[b][0] == 0 || shapes_a[b][1] == 0 || shapes_a[b][2] == 0) continue;
        const int64_t start_x = starts_a[b][0];
        const int64_t start_y = starts_a[b][1];
        const int64_t start_z = starts_a[b][2];
        const int64_t size_x = shapes_a[b][0];
        const int64_t size_y = shapes_a[b][1];
        const int64_t size_z = shapes_a[b][2];
        for (const auto& ck : brick_keys[b]) {
            int64_t lx = ck.x - start_x;
            int64_t ly = ck.y - start_y;
            int64_t lz = ck.z - start_z;
            if (lx < 0 || ly < 0 || lz < 0 || lx >= size_x || ly >= size_y || lz >= size_z) continue;
            for (int i = 0; i < 8; ++i) {
                VoxelKey vk{ck.x + dx[i], ck.y + dy[i], ck.z + dz[i]};
                auto it = key_to_row.find(vk);
                if (it == key_to_row.end()) continue;
                int64_t vx = vk.x - start_x;
                int64_t vy = vk.y - start_y;
                int64_t vz = vk.z - start_z;
                if (vx < 0 || vy < 0 || vz < 0 || vx > size_x || vy > size_y || vz > size_z) continue;
                int64_t idx = vert_index(b, vx, vy, vz);
                const int64_t r = it->second;
                sig_ptr[idx] = feats_acc[r][0];
                col_ptr[idx * 3 + 0] = feats_acc[r][1];
                col_ptr[idx * 3 + 1] = feats_acc[r][2];
                col_ptr[idx * 3 + 2] = feats_acc[r][3];
                val_ptr[idx] = 1;
            }
        }
    }

    if (on_cuda) {
        auto device = torch::kCUDA;
        return {
            sigma_cpu.to(device),
            color_cpu.to(device),
            valid_cpu.to(device),
            brick_starts.to(device),
            brick_shapes.to(device),
            brick_vertex_offsets.to(device),
        };
    }
    return {sigma_cpu, color_cpu, valid_cpu, brick_starts, brick_shapes, brick_vertex_offsets};
}

torch::Tensor rasterize_image(
    VoxelGrid& grid,
    const torch::Tensor& intrinsic,
    const torch::Tensor& extrinsic,
    int64_t height,
    int64_t width,
    int64_t coarse_res,
    const torch::Tensor& vertex_features) {
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
    auto dense_vertices = build_dense_vertex_grids_from_features(grid, vertex_features);
    auto vertex_sigma = std::get<0>(dense_vertices);
    auto vertex_color = std::get<1>(dense_vertices);
    auto vertex_mask = std::get<2>(dense_vertices);

    // 相机参数 -> 射线
    auto rays = build_rays_from_camera(intrinsic, extrinsic, height, width, device);
    torch::Tensor rays_o = rays.first.contiguous();
    torch::Tensor rays_d = rays.second.contiguous();

    const bool has_vertex = vertex_sigma.defined() && vertex_color.defined();
    if (has_vertex) {
        auto dev = rays_o.device();
        vertex_sigma = vertex_sigma.to(dev).contiguous();
        vertex_color = vertex_color.to(dev).contiguous();
        if (vertex_mask.defined() && vertex_mask.numel() > 0) {
            vertex_mask = vertex_mask.to(dev).contiguous();
        }
    }

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
        rays_o, rays_d, offsets, keys, mask, vertex_sigma, vertex_color, vertex_mask,
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
    int64_t coarse_res,
    const torch::Tensor& vertex_features) {
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

    auto dense_vertices = build_dense_vertex_grids_from_features(grid, vertex_features);
    auto vertex_sigma = std::get<0>(dense_vertices);
    auto vertex_color = std::get<1>(dense_vertices);
    auto vertex_mask = std::get<2>(dense_vertices);
    const bool has_vertex = vertex_sigma.defined() && vertex_color.defined();
    if (has_vertex) {
        auto dev = rays_o.device();
        vertex_sigma = vertex_sigma.to(dev).contiguous();
        vertex_color = vertex_color.to(dev).contiguous();
        if (vertex_mask.defined() && vertex_mask.numel() > 0) {
            vertex_mask = vertex_mask.to(dev).contiguous();
        }
    }

    return rasterize_forward_cuda(
        rays_o, rays_d, offsets, keys, mask,
        vertex_sigma, vertex_color, vertex_mask,
        height, width,
        static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(origin.z),
        static_cast<float>(voxel_size),
        dims.x, dims.y, dims.z,
        brick_x, brick_y, brick_z,
        coarse_res);
}

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
    int64_t coarse_res) {
    check_matrix(intrinsic, 3, 3, "intrinsic");
    check_matrix(extrinsic, 4, 4, "extrinsic");
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is required for rasterize_image_with_index_dense");
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

    auto v_sigma = vertex_sigma.to(device).contiguous().to(torch::kFloat32);
    auto v_color = vertex_color.to(device).contiguous().to(torch::kFloat32);
    auto v_mask = vertex_mask.defined() && vertex_mask.numel() > 0
        ? vertex_mask.to(device).contiguous()
        : torch::Tensor();

    return rasterize_forward_cuda(
        rays_o, rays_d, offsets, keys, mask,
        v_sigma, v_color, v_mask,
        height, width,
        static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(origin.z),
        static_cast<float>(voxel_size),
        dims.x, dims.y, dims.z,
        brick_x, brick_y, brick_z,
        coarse_res);
}

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
    int64_t coarse_res) {
    check_matrix(intrinsic, 3, 3, "intrinsic");
    check_matrix(extrinsic, 4, 4, "extrinsic");
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is required for rasterize_image_with_index_packed");
    }
    torch::Device device(torch::kCUDA);

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

    auto offsets = coarse_offsets.to(device).contiguous();
    auto keys = voxel_keys.to(device).contiguous();
    auto mask = coarse_mask.to(device).contiguous();

    auto v_sigma = vertex_sigma_packed.to(device).contiguous().to(torch::kFloat32);
    auto v_color = vertex_color_packed.to(device).contiguous().to(torch::kFloat32);
    auto v_mask = (vertex_mask_packed.defined() && vertex_mask_packed.numel() > 0)
        ? vertex_mask_packed.to(device).contiguous()
        : torch::Tensor();

    auto b_starts = brick_starts.to(device).contiguous();
    auto b_shapes = brick_shapes.to(device).contiguous();
    auto b_offsets = brick_vertex_offsets.to(device).contiguous();

    return rasterize_forward_packed_cuda(
        rays_o, rays_d, offsets, keys, mask,
        v_sigma, v_color, v_mask,
        b_starts, b_shapes, b_offsets,
        height, width,
        static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(origin.z),
        static_cast<float>(voxel_size),
        dims.x, dims.y, dims.z,
        brick_x, brick_y, brick_z,
        coarse_res);
}

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
    int64_t coarse_res) {
    check_matrix(intrinsic, 3, 3, "intrinsic");
    check_matrix(extrinsic, 4, 4, "extrinsic");
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is required for rasterize_image_with_index_dense_backward");
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
        throw std::runtime_error("grid dims are zero; cannot rasterize backward");
    }
    int64_t brick_x = static_cast<int64_t>(std::ceil(static_cast<double>(dims.x) / coarse_res));
    int64_t brick_y = static_cast<int64_t>(std::ceil(static_cast<double>(dims.y) / coarse_res));
    int64_t brick_z = static_cast<int64_t>(std::ceil(static_cast<double>(dims.z) / coarse_res));

    // 将预计算的 coarse 索引搬到 GPU
    auto offsets = coarse_offsets.to(device).contiguous();
    auto keys = voxel_keys.to(device).contiguous();
    auto mask = coarse_mask.to(device).contiguous();

    auto v_sigma = vertex_sigma.to(device).contiguous().to(torch::kFloat32);
    auto v_color = vertex_color.to(device).contiguous().to(torch::kFloat32);
    auto v_mask = vertex_mask.defined() && vertex_mask.numel() > 0
        ? vertex_mask.to(device).contiguous()
        : torch::Tensor();
    auto gout = grad_output.to(device).contiguous().to(torch::kFloat32);

    return rasterize_backward_cuda(
        rays_o, rays_d, offsets, keys, mask,
        v_sigma, v_color, v_mask, gout,
        height, width,
        static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(origin.z),
        static_cast<float>(voxel_size),
        dims.x, dims.y, dims.z,
        brick_x, brick_y, brick_z,
        coarse_res);
}

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
    int64_t coarse_res) {
    check_matrix(intrinsic, 3, 3, "intrinsic");
    check_matrix(extrinsic, 4, 4, "extrinsic");
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is required for rasterize_image_with_index_packed_backward");
    }
    torch::Device device(torch::kCUDA);

    auto rays = build_rays_from_camera(intrinsic, extrinsic, height, width, device);
    torch::Tensor rays_o = rays.first.contiguous();
    torch::Tensor rays_d = rays.second.contiguous();

    Vec3 origin = grid.origin();
    Vec3i dims = grid.dims();
    double voxel_size = grid.voxel_size();
    if (dims.x <= 0 || dims.y <= 0 || dims.z <= 0) {
        throw std::runtime_error("grid dims are zero; cannot rasterize backward");
    }
    int64_t brick_x = static_cast<int64_t>(std::ceil(static_cast<double>(dims.x) / coarse_res));
    int64_t brick_y = static_cast<int64_t>(std::ceil(static_cast<double>(dims.y) / coarse_res));
    int64_t brick_z = static_cast<int64_t>(std::ceil(static_cast<double>(dims.z) / coarse_res));

    auto offsets = coarse_offsets.to(device).contiguous();
    auto keys = voxel_keys.to(device).contiguous();
    auto mask = coarse_mask.to(device).contiguous();

    auto v_sigma = vertex_sigma_packed.to(device).contiguous().to(torch::kFloat32);
    auto v_color = vertex_color_packed.to(device).contiguous().to(torch::kFloat32);
    auto v_mask = (vertex_mask_packed.defined() && vertex_mask_packed.numel() > 0)
        ? vertex_mask_packed.to(device).contiguous()
        : torch::Tensor();
    auto b_starts = brick_starts.to(device).contiguous();
    auto b_shapes = brick_shapes.to(device).contiguous();
    auto b_offsets = brick_vertex_offsets.to(device).contiguous();
    auto gout = grad_output.to(device).contiguous().to(torch::kFloat32);

    return rasterize_backward_packed_cuda(
        rays_o, rays_d, offsets, keys, mask,
        v_sigma, v_color, v_mask,
        b_starts, b_shapes, b_offsets,
        gout,
        height, width,
        static_cast<float>(origin.x), static_cast<float>(origin.y), static_cast<float>(origin.z),
        static_cast<float>(voxel_size),
        dims.x, dims.y, dims.z,
        brick_x, brick_y, brick_z,
        coarse_res);
}
