#pragma once

#include <unordered_map>
#include <vector>
#include <string>

#include "voxel_node.h"

// 稀疏体素网格，支持共享顶点和占据体素的稀疏存储。
class VoxelGrid {
public:
    VoxelGrid() = default;
    ~VoxelGrid() = default;

    // 基于原点、尺寸和体素尺寸构建致密网格（随后可稀疏化）。
    void build_dense(const Vec3& origin, const Vec3& size, double voxel_size);

    // 插入一个体素（若不存在），返回指向体素的指针。
    VoxelCell* ensure_cell(const VoxelKey& key);

    size_t cell_count() const { return cells_.size(); }
    size_t vertex_count() const { return vertices_.size(); }

    const std::unordered_map<VoxelKey, VoxelCell, VoxelKeyHash>& cells() const { return cells_; }
    const std::unordered_map<VoxelKey, std::shared_ptr<Vertex>, VoxelKeyHash>& vertices() const { return vertices_; }

    // 获取体素的 8 个顶点 key（若不存在返回空列表）
    std::vector<VoxelKey> cell_corner_keys(const VoxelKey& key) const;

    // 将孤立顶点清理（未被任何体素引用）
    void gc_vertices();

    // 网格元信息
    Vec3 origin() const { return origin_; }
    Vec3 size() const { return size_; }
    double voxel_size() const { return voxel_size_; }
    Vec3i dims() const { return dims_; }
    void set_metadata(const Vec3& origin, const Vec3& size, double voxel_size, const Vec3i& dims) {
        origin_ = origin; size_ = size; voxel_size_ = voxel_size; dims_ = dims;
    }

    // 导出所有体素的四面体网格（共享顶点映射到全局表）
    TetraMesh to_tetra_mesh() const;
    void write_tetra_ply(const std::string& path) const;

private:
    std::shared_ptr<Vertex> get_or_create_vertex(const VoxelKey& key);

    std::unordered_map<VoxelKey, std::shared_ptr<Vertex>, VoxelKeyHash> vertices_;
    std::unordered_map<VoxelKey, VoxelCell, VoxelKeyHash> cells_;

    Vec3 origin_{0.0, 0.0, 0.0};
    Vec3 size_{0.0, 0.0, 0.0};
    double voxel_size_{1.0};
    Vec3i dims_{0, 0, 0};
};
