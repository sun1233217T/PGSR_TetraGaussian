#include "voxel_initialize.h"

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
