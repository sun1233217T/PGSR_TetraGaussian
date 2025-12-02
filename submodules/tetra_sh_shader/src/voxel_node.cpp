#include "voxel_node.h"

#include <algorithm>
#include <fstream>
#include <unordered_set>

namespace {
struct FaceHash {
    size_t operator()(const std::array<int,3>& f) const noexcept {
        size_t h = static_cast<size_t>(f[0]);
        h ^= static_cast<size_t>(f[1]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(f[2]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};
} // namespace

void TetraMesh::write_ply(const std::string& path) const {
    // 收集三角面并去重
    std::vector<std::array<int,3>> faces;
    std::unordered_set<std::array<int,3>, FaceHash> seen;
    auto add_face = [&](int a, int b, int c) {
        std::array<int,3> key{a,b,c};
        std::array<int,3> sorted = key;
        std::sort(sorted.begin(), sorted.end());
        if (seen.insert(sorted).second) faces.push_back(key);
    };
    for (const auto& t : tets) {
        add_face(t[0], t[1], t[2]);
        add_face(t[0], t[2], t[3]);
        add_face(t[0], t[3], t[1]);
        add_face(t[1], t[3], t[2]);
    }

    std::ofstream ofs(path, std::ios::out);
    if (!ofs) return;

    ofs << "ply\nformat ascii 1.0\n";
    ofs << "element vertex " << vertices.size() << "\n";
    ofs << "property double x\nproperty double y\nproperty double z\n";
    ofs << "element face " << faces.size() << "\n";
    ofs << "property list uchar int vertex_indices\n";
    ofs << "end_header\n";
    for (const auto& v : vertices) {
        ofs << v.x << " " << v.y << " " << v.z << "\n";
    }
    for (const auto& f : faces) {
        ofs << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
}

TetraMesh VoxelCell::to_tetra_mesh(const Vec3& origin, double voxel_size) const {
    TetraMesh m;
    m.vertices.reserve(8);
    for (const auto& v : corners) {
        Vec3 pos{
            origin.x + static_cast<double>(v->key.x) * voxel_size,
            origin.y + static_cast<double>(v->key.y) * voxel_size,
            origin.z + static_cast<double>(v->key.z) * voxel_size
        };
        m.vertices.push_back(pos);
    }
    // 角点顺序与 ensure_cell 中一致：0(000),1(100),2(010),3(110),4(001),5(101),6(011),7(111)
    // 使用对角线 0->6 的经典 6 四面体划分
    // [0, 1, 3, 5],  # Tetrahedron 1
    // [0, 4, 5, 7],  # Tetrahedron 2
    // [0, 3, 5, 7],  # Tetrahedron 3
    // [0, 2, 3, 7],  # Tetrahedron 4
    // [0, 2, 4, 7],  # Tetrahedron 5
    // [2, 4, 6, 7],  # Tetrahedron 6
    m.tets = {
        {0, 1, 3, 5},
        {0, 4, 5, 7},
        {0, 3, 5, 7},
        {0, 2, 3, 7},
        {0, 2, 4, 7},
        {2, 4, 6, 7}
    };
    return m;
}

void VoxelCell::write_tetra_ply(const Vec3& origin, double voxel_size, const std::string& path) const {
    auto mesh = to_tetra_mesh(origin, voxel_size);
    mesh.write_ply(path);
}
