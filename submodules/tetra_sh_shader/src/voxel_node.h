#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>

struct Vec3 {
    double x{0.0}, y{0.0}, z{0.0};
    Vec3() = default;
    Vec3(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {}
};

struct Vec3i {
    int64_t x{0}, y{0}, z{0};
    Vec3i() = default;
    Vec3i(int64_t xx, int64_t yy, int64_t zz) : x(xx), y(yy), z(zz) {}
};

struct VoxelKey {
    int64_t x{0}, y{0}, z{0};
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const noexcept {
        // 简单的 3D Morton-like 混合
        uint64_t h = static_cast<uint64_t>(k.x) * 0x9e3779b97f4a7c15ULL;
        h ^= static_cast<uint64_t>(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<uint64_t>(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return static_cast<size_t>(h);
    }
};

// 顶点属性基类：留空，方便后续扩展/从 Python 构造
class Attribute {
public:
    Attribute() = default;
    virtual ~Attribute() = default;
};

class IntAttribute : public Attribute {
public:
    int value{0};
    IntAttribute() = default;
    explicit IntAttribute(int v) : value(v) {}
};

// 网格顶点，包含整数格点坐标和共享的属性指针
class Vertex {
public:
    VoxelKey key;
    std::shared_ptr<Attribute> attr;

    Vertex() = default;
    explicit Vertex(const VoxelKey& k) : key(k) {}
};

struct TetraMesh {
    std::vector<Vec3> vertices;
    std::vector<std::array<int, 4>> tets;  // 四面体由 4 个顶点索引组成

    // 写出为 ASCII PLY，三角面去重
    void write_ply(const std::string& path) const;
};

// 单个体素单元，保存 8 个共享顶点
class VoxelCell {
public:
    VoxelKey key;  // 体素索引（整数格点为 min corner）
    std::array<std::shared_ptr<Vertex>, 8> corners;

    // 使用单元 8 个角点生成四面体划分
    TetraMesh to_tetra_mesh(const Vec3& origin, double voxel_size) const;
    void write_tetra_ply(const Vec3& origin, double voxel_size, const std::string& path) const;
};
