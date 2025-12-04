#include "voxel_grid.h"

#include <cmath>
#include <algorithm>
#include <iostream>

std::shared_ptr<Vertex> VoxelGrid::get_or_create_vertex(const VoxelKey& key) {
    auto it = vertices_.find(key);
    if (it != vertices_.end()) return it->second;
    auto v = std::make_shared<Vertex>(key);
    vertices_.emplace(key, v);
    return v;
}

VoxelCell* VoxelGrid::ensure_cell(const VoxelKey& key) {
    auto it = cells_.find(key);
    if (it != cells_.end()) return &it->second;

    VoxelCell cell;
    cell.key = key;

    // 8 corner offsets relative to the cell min corner.
    const int dx[8] = {0,1,0,1,0,1,0,1};
    const int dy[8] = {0,0,1,1,0,0,1,1};
    const int dz[8] = {0,0,0,0,1,1,1,1};
    for (int i = 0; i < 8; ++i) {
        VoxelKey vk{key.x + dx[i], key.y + dy[i], key.z + dz[i]};
        cell.corners[i] = get_or_create_vertex(vk);
    }

    auto res = cells_.emplace(key, std::move(cell));
    return &res.first->second;
}

void VoxelGrid::build_dense(const Vec3& origin, const Vec3& size, double voxel_size) {
    if (voxel_size <= 0.0) return;
    cells_.clear();
    vertices_.clear();

    origin_ = origin;
    size_ = size;
    voxel_size_ = voxel_size;

    const int64_t nx = static_cast<int64_t>(std::ceil(size.x / voxel_size));
    const int64_t ny = static_cast<int64_t>(std::ceil(size.y / voxel_size));
    const int64_t nz = static_cast<int64_t>(std::ceil(size.z / voxel_size));
    dims_ = Vec3i(nx, ny, nz);
    std::cout << "VoxelGrid::build_dense: dims = (" << nx << ", " << ny << ", " << nz << ")" << std::endl;

    for (int64_t ix = 0; ix < nx; ++ix) {
        for (int64_t iy = 0; iy < ny; ++iy) {
            for (int64_t iz = 0; iz < nz; ++iz) {
                VoxelKey key{ix, iy, iz};
                ensure_cell(key);
            }
        }
    }
    (void)origin; // origin 暂未用于索引定位，后续可加物理坐标接口
}

std::vector<VoxelKey> VoxelGrid::cell_corner_keys(const VoxelKey& key) const {
    std::vector<VoxelKey> out;
    auto it = cells_.find(key);
    if (it == cells_.end()) return out;
    out.reserve(8);
    for (const auto& v : it->second.corners) {
        out.push_back(v->key);
    }
    return out;
}

void VoxelGrid::gc_vertices() {
    // 统计引用
    std::unordered_map<VoxelKey, size_t, VoxelKeyHash> refcnt;
    for (const auto& kv : cells_) {
        for (const auto& v : kv.second.corners) {
            refcnt[v->key] += 1;
        }
    }
    // 删除未引用顶点
    for (auto it = vertices_.begin(); it != vertices_.end(); ) {
        if (refcnt.find(it->first) == refcnt.end()) {
            it = vertices_.erase(it);
        } else {
            ++it;
        }
    }
}

TetraMesh VoxelGrid::to_tetra_mesh() const {
    TetraMesh mesh;
    mesh.vertices.reserve(vertices_.size());
    mesh.tets.reserve(cells_.size() * 6);

    // 构建全局顶点索引表（按 key 排序保证稳定）
    std::vector<VoxelKey> keys;
    keys.reserve(vertices_.size());
    for (const auto& kv : vertices_) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end(), [](const VoxelKey& a, const VoxelKey& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    });
    std::unordered_map<VoxelKey, int, VoxelKeyHash> key_to_idx;
    key_to_idx.reserve(keys.size());
    for (int idx = 0; idx < static_cast<int>(keys.size()); ++idx) {
        key_to_idx[keys[idx]] = idx;
        mesh.vertices.push_back(Vec3{
            origin_.x + static_cast<double>(keys[idx].x) * voxel_size_,
            origin_.y + static_cast<double>(keys[idx].y) * voxel_size_,
            origin_.z + static_cast<double>(keys[idx].z) * voxel_size_
        });
    }

    auto push_cell_tets = [&](const VoxelCell& cell) {
        int ids[8];
        for (int i = 0; i < 8; ++i) {
            ids[i] = key_to_idx.at(cell.corners[i]->key);
        }
        mesh.tets.push_back({ids[0], ids[1], ids[3], ids[5]});
        mesh.tets.push_back({ids[0], ids[4], ids[5], ids[7]});
        mesh.tets.push_back({ids[0], ids[3], ids[5], ids[7]});
        mesh.tets.push_back({ids[0], ids[2], ids[3], ids[7]});
        mesh.tets.push_back({ids[0], ids[2], ids[4], ids[7]});
        mesh.tets.push_back({ids[2], ids[4], ids[6], ids[7]});
    };

    for (const auto& kv : cells_) {
        push_cell_tets(kv.second);
    }
    return mesh;
}

void VoxelGrid::write_tetra_ply(const std::string& path) const {
    auto m = to_tetra_mesh();
    m.write_ply(path);
}
