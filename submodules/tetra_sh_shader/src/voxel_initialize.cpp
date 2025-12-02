#include "voxel_initialize.h"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "voxel_utility.h"

namespace {

// torch 行视图属性：持有整块张量和当前行索引，避免拷贝。
class TensorAttribute : public Attribute {
public:
    TensorAttribute(std::shared_ptr<torch::Tensor> backing, int64_t row)
        : tensor_(std::move(backing)), row_(row), dim_(tensor_ ? (*tensor_).size(1) : 0) {}

    torch::Tensor view() const {
        return tensor_ ? tensor_->select(0, row_) : torch::Tensor();
    }
    int64_t row() const { return row_; }
    int64_t dim() const { return dim_; }
    const torch::Tensor& tensor() const { return *tensor_; }

private:
    std::shared_ptr<torch::Tensor> tensor_;
    int64_t row_{0};
    int64_t dim_{0};
};

torch::TensorOptions make_options(bool on_cuda, bool as_int) {
    auto device = on_cuda ? torch::kCUDA : torch::kCPU;
    auto dtype = as_int ? torch::kInt64 : torch::kFloat32;
    return torch::TensorOptions().device(device).dtype(dtype);
}

} // namespace

torch::Tensor init_vertex_tensor(VoxelGrid& grid,
                                 int64_t feature_dim,
                                 bool on_cuda,
                                 bool fill_ones,
                                 bool as_int) {
    if (feature_dim <= 0) throw std::invalid_argument("feature_dim must be positive");

    auto ordered = sorted_vertices(grid);
    const int64_t N = static_cast<int64_t>(ordered.size());

    auto options = make_options(on_cuda, as_int);
    const double fill_val = fill_ones ? 1.0 : 0.0;
    torch::Tensor tensor = torch::full({N, feature_dim}, fill_val, options);
    auto backing = std::make_shared<torch::Tensor>(tensor);

    int64_t row = 0;
    for (const auto& kv : ordered) {
        kv.second->attr = std::make_shared<TensorAttribute>(backing, row);
        ++row;
    }
    return tensor;
}

torch::Tensor attach_vertex_tensor(VoxelGrid& grid, const torch::Tensor& tensor_in) {
    if (tensor_in.dim() != 2) {
        throw std::invalid_argument("tensor must be 2D (N, C)");
    }
    auto tensor = tensor_in.contiguous();
    auto ordered = sorted_vertices(grid);
    const int64_t N = static_cast<int64_t>(ordered.size());
    if (tensor.size(0) != N) {
        throw std::invalid_argument("tensor first dimension does not match vertex count");
    }
    auto backing = std::make_shared<torch::Tensor>(tensor);

    int64_t row = 0;
    for (const auto& kv : ordered) {
        kv.second->attr = std::make_shared<TensorAttribute>(backing, row);
        ++row;
    }
    return tensor;
}

size_t ensure_active_cells_from_vertex_attr(VoxelGrid& grid) {
    const Vec3i dims = grid.dims();
    const size_t before = grid.cell_count();

    for (const auto& kv : grid.vertices()) {
        const auto& v = kv.second;
        if (!v->attr) continue;
        const int64_t vx = v->key.x;
        const int64_t vy = v->key.y;
        const int64_t vz = v->key.z;
        for (int dx : {0, -1}) {
            for (int dy : {0, -1}) {
                for (int dz : {0, -1}) {
                    Vec3i idx{vx + dx, vy + dy, vz + dz};
                    if (!cell_index_inside(idx, dims)) continue;
                    VoxelKey ck{idx.x, idx.y, idx.z};
                    grid.ensure_cell(ck);
                }
            }
        }
    }
    const size_t after = grid.cell_count();
    return (after >= before) ? (after - before) : 0;
}

torch::Tensor build_coarse_occupancy(const VoxelGrid& grid, int64_t res, bool on_cuda) {
    if (res <= 0) throw std::invalid_argument("coarse resolution must be positive");
    torch::Tensor coarse_cpu = torch::zeros({res, res, res},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));

    // 使用物理尺寸映射到 coarse 网格
    const Vec3 origin = grid.origin();
    const Vec3 size = grid.size();
    const double vx = grid.voxel_size();

    auto acc = coarse_cpu.accessor<uint8_t, 3>();
    for (const auto& kv : grid.cells()) {
        const VoxelKey& ck = kv.first;
        // 体素最小角对应的世界坐标
        const double px = origin.x + static_cast<double>(ck.x) * vx;
        const double py = origin.y + static_cast<double>(ck.y) * vx;
        const double pz = origin.z + static_cast<double>(ck.z) * vx;
        auto to_c = [&](double p, double s, double o) -> int64_t {
            if (s <= 0.0) return 0;
            double u = (p - o) / s; // origin 到该轴起点的归一化
            int64_t idx = static_cast<int64_t>(std::floor(u * res));
            if (idx < 0) idx = 0;
            if (idx >= res) idx = res - 1;
            return idx;
        };
        const int64_t cx = to_c(px, size.x, origin.x);
        const int64_t cy = to_c(py, size.y, origin.y);
        const int64_t cz = to_c(pz, size.z, origin.z);
        acc[cx][cy][cz] = 1;
    }
    if (on_cuda) {
        return coarse_cpu.to(torch::kCUDA);
    }
    return coarse_cpu;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_coarse_index(const VoxelGrid& grid, int64_t res, bool on_cuda) {
    if (res <= 0) throw std::invalid_argument("coarse resolution must be positive");

    const int64_t B = res * res * res;
    const Vec3i dims = grid.dims();
    if (dims.x <= 0 || dims.y <= 0 || dims.z <= 0) {
        torch::Tensor occ = torch::zeros({res, res, res}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        torch::Tensor offsets = torch::zeros({B + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        torch::Tensor keys = torch::zeros({0, 3}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        torch::Tensor mask = torch::zeros({B}, torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCPU));
        if (on_cuda) {
            occ = occ.to(torch::kCUDA);
            offsets = offsets.to(torch::kCUDA);
            keys = keys.to(torch::kCUDA);
            mask = mask.to(torch::kCUDA);
        }
        return {occ, offsets, keys, mask};
    }

    const int64_t brick_size_x = static_cast<int64_t>(std::ceil(static_cast<double>(dims.x) / res));
    const int64_t brick_size_y = static_cast<int64_t>(std::ceil(static_cast<double>(dims.y) / res));
    const int64_t brick_size_z = static_cast<int64_t>(std::ceil(static_cast<double>(dims.z) / res));

    std::vector<int64_t> counts(B, 0);
    for (const auto& kv : grid.cells()) {
        const VoxelKey& ck = kv.first;
        int64_t bx = ck.x / brick_size_x;
        int64_t by = ck.y / brick_size_y;
        int64_t bz = ck.z / brick_size_z;
        if (bx >= res) bx = res - 1;
        if (by >= res) by = res - 1;
        if (bz >= res) bz = res - 1;
        const int64_t b = (bx * res + by) * res + bz;
        counts[b] += 1;
    }

    torch::Tensor occ_cpu = torch::zeros({res, res, res}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    torch::Tensor offsets_cpu = torch::zeros({B + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    torch::Tensor mask_cpu = torch::zeros({B}, torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCPU));

    // prefix sum
    auto offsets_acc = offsets_cpu.accessor<int64_t, 1>();
    int64_t total = 0;
    for (int64_t b = 0; b < B; ++b) {
        offsets_acc[b] = total;
        total += counts[b];
        if (counts[b] > 0) {
            int64_t bx = b / (res * res);
            int64_t by = (b / res) % res;
            int64_t bz = b % res;
            occ_cpu.index_put_({bx, by, bz}, 1);
        }
    }
    offsets_acc[B] = total;

    torch::Tensor keys_cpu = torch::zeros({total, 3}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    std::vector<int64_t> write_ptr = counts;
    for (int64_t& v : write_ptr) v = 0;

    auto keys_acc = keys_cpu.accessor<int64_t, 2>();
    auto mask_acc = mask_cpu.accessor<uint64_t, 1>();

    for (const auto& kv : grid.cells()) {
        const VoxelKey& ck = kv.first;
        int64_t bx = ck.x / brick_size_x;
        int64_t by = ck.y / brick_size_y;
        int64_t bz = ck.z / brick_size_z;
        if (bx >= res) bx = res - 1;
        if (by >= res) by = res - 1;
        if (bz >= res) bz = res - 1;
        const int64_t b = (bx * res + by) * res + bz;

        int64_t idx = offsets_acc[b] + write_ptr[b];
        write_ptr[b] += 1;
        keys_acc[idx][0] = ck.x;
        keys_acc[idx][1] = ck.y;
        keys_acc[idx][2] = ck.z;

        // 砖内粗 bitmask：将体素归一化到 4x4x4 子格并设位（碰撞只会增加冗余）
        auto q = [](int64_t c, int64_t bs) -> int64_t {
            if (bs <= 0) return 0;
            double frac = static_cast<double>(c % bs) / static_cast<double>(bs);
            int64_t v = static_cast<int64_t>(std::floor(frac * 4.0));
            if (v < 0) v = 0;
            if (v > 3) v = 3;
            return v;
        };
        int64_t lx = q(ck.x, brick_size_x);
        int64_t ly = q(ck.y, brick_size_y);
        int64_t lz = q(ck.z, brick_size_z);
        int64_t bit = (lx << 4) | (ly << 2) | lz; // 6 bits => 64 slots
        mask_acc[b] |= (1ULL << bit);
    }

    if (on_cuda) {
        auto device = torch::kCUDA;
        return {
            occ_cpu.to(device),
            offsets_cpu.to(device),
            keys_cpu.to(device),
            mask_cpu.to(device)
        };
    }
    return {occ_cpu, offsets_cpu, keys_cpu, mask_cpu};
}
