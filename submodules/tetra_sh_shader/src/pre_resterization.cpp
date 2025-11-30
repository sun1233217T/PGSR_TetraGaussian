#include "pre_resterization.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <limits>
#include <unordered_set>
#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

inline bool inside(const Vec3i& idx, const Vec3i& dims) {
    return idx.x >= 0 && idx.y >= 0 && idx.z >= 0 &&
           idx.x < dims.x && idx.y < dims.y && idx.z < dims.z;
}

// 3D DDA 遍历，与轴对齐体素网格交点。
void traverse_ray(const Vec3& o, const Vec3& d, const VoxelGrid& grid,
                  std::function<void(const VoxelKey&)> visit) {
    Vec3 minb = grid.origin();
    Vec3 maxb{grid.origin().x + grid.size().x,
              grid.origin().y + grid.size().y,
              grid.origin().z + grid.size().z};

    auto slab = [&](double o_c, double d_c, double min_c, double max_c, double& t0, double& t1) {
        if (std::abs(d_c) < 1e-12) {
            if (o_c < min_c || o_c > max_c) return false;
            return true;
        }
        double tmin = (min_c - o_c) / d_c;
        double tmax = (max_c - o_c) / d_c;
        if (tmin > tmax) std::swap(tmin, tmax);
        t0 = std::max(t0, tmin);
        t1 = std::min(t1, tmax);
        return t0 <= t1;
    };

    double t0 = 0.0, t1 = std::numeric_limits<double>::infinity();
    if (!slab(o.x, d.x, minb.x, maxb.x, t0, t1)) return;
    if (!slab(o.y, d.y, minb.y, maxb.y, t0, t1)) return;
    if (!slab(o.z, d.z, minb.z, maxb.z, t0, t1)) return;
    if (t1 < 0.0) return; // 整条射线在盒子之前
    if (t0 < 0.0) t0 = 0.0;

    const double vs = grid.voxel_size();
    Vec3 p = {o.x + t0 * d.x, o.y + t0 * d.y, o.z + t0 * d.z};
    Vec3 rel = { (p.x - minb.x) / vs, (p.y - minb.y) / vs, (p.z - minb.z) / vs };
    Vec3i idx{static_cast<int64_t>(std::floor(rel.x)),
              static_cast<int64_t>(std::floor(rel.y)),
              static_cast<int64_t>(std::floor(rel.z))};

    Vec3i step{ (d.x > 0) ? 1 : (d.x < 0 ? -1 : 0),
                (d.y > 0) ? 1 : (d.y < 0 ? -1 : 0),
                (d.z > 0) ? 1 : (d.z < 0 ? -1 : 0) };

    auto t_max_axis = [&](double o_c, double d_c, int64_t idx_c, double min_c) {
        if (std::abs(d_c) < 1e-12) return std::numeric_limits<double>::infinity();
        double next_boundary = min_c + ( (d_c > 0 ? idx_c + 1 : idx_c) ) * vs;
        return (next_boundary - o_c) / d_c;
    };
    auto t_delta_axis = [&](double d_c) {
        if (std::abs(d_c) < 1e-12) return std::numeric_limits<double>::infinity();
        return vs / std::abs(d_c);
    };

    double tMaxX = t_max_axis(o.x, d.x, idx.x, minb.x);
    double tMaxY = t_max_axis(o.y, d.y, idx.y, minb.y);
    double tMaxZ = t_max_axis(o.z, d.z, idx.z, minb.z);
    double tDeltaX = t_delta_axis(d.x);
    double tDeltaY = t_delta_axis(d.y);
    double tDeltaZ = t_delta_axis(d.z);

    Vec3i dims = grid.dims();
    while (inside(idx, dims)) {
        visit(VoxelKey{idx.x, idx.y, idx.z});
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                idx.x += step.x; tMaxX += tDeltaX;
            } else {
                idx.z += step.z; tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                idx.y += step.y; tMaxY += tDeltaY;
            } else {
                idx.z += step.z; tMaxZ += tDeltaZ;
            }
        }
        double tMin = std::min({tMaxX, tMaxY, tMaxZ});
        if (tMin > t1) break;
    }
}

} // namespace

void grid_init(VoxelGrid& grid) {
    for (auto& kv : grid.vertices()) {
        kv.second->attr = std::make_shared<IntAttribute>(1);
    }
}

void image_pre_resterization(VoxelGrid& grid, const pybind11::array& origins_in, const pybind11::array& dirs_in) {
    // origins: (N,3), dirs: (N,M,3) 或 (K,3)
    // 强制转 double，避免 float32 被当成 double 读取导致数据错位。
    py::array origins = py::array_t<double, py::array::c_style | py::array::forcecast>(origins_in);
    py::array dirs     = py::array_t<double, py::array::c_style | py::array::forcecast>(dirs_in);

    py::buffer_info oinfo = origins.request();
    py::buffer_info dinfo = dirs.request();
    if (oinfo.ndim != 2 || oinfo.shape[1] != 3) throw std::invalid_argument("origins 形状应为 (N,3)");
    if (!(dinfo.ndim == 2 || dinfo.ndim == 3) || dinfo.shape[dinfo.ndim-1] != 3)
        throw std::invalid_argument("dirs 形状应为 (K,3) 或 (N,M,3)");
    std::cout << "Starting pre-rasterization with "
              << oinfo.shape[0] << " origins and "
              << ((dinfo.ndim == 3) ? dinfo.shape[1] : dinfo.shape[0]) << " directions each."
              << std::endl;

    auto load_vec = [](const py::buffer_info& info, py::ssize_t i0, py::ssize_t i1) -> Vec3 {
        const double* data = static_cast<const double*>(info.ptr);
        auto stride0 = info.strides[0] / static_cast<py::ssize_t>(sizeof(double));
        auto stride1 = info.strides[1] / static_cast<py::ssize_t>(sizeof(double));
        const double* row = data + i0 * stride0 + i1 * stride1;
        return Vec3{row[0], row[1], row[2]};
    };

    py::ssize_t N = oinfo.shape[0];
    py::ssize_t M = (dinfo.ndim == 3) ? dinfo.shape[1] : dinfo.shape[0];
    const py::ssize_t total = N * M;
    const bool debug_rays = (total <= 4);
    const py::ssize_t progress_step = std::max<py::ssize_t>(1, total / 100);

    auto ray_accessor = [&](py::ssize_t idx) {
        py::ssize_t i = idx / M;
        py::ssize_t j = idx % M;
        Vec3 o = load_vec(oinfo, i, 0);
        Vec3 d;
        if (dinfo.ndim == 3) d = load_vec(dinfo, i, j);
        else d = load_vec(dinfo, j, 0);
        return std::pair<Vec3, Vec3>(o, d);
    };

    // 并行收集命中体素键
#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
    std::vector<std::unordered_set<VoxelKey, VoxelKeyHash>> local_sets(n_threads);
    std::atomic<py::ssize_t> done{0};
#pragma omp parallel for schedule(static)
    for (py::ssize_t idx = 0; idx < total; ++idx) {
        auto [o, d] = ray_accessor(idx);
        auto& visited = local_sets[omp_get_thread_num()];
        std::vector<VoxelKey> debug_hits;
        traverse_ray(o, d, grid, [&](const VoxelKey& key){
            visited.insert(key);
            if (debug_rays && idx < 4) debug_hits.push_back(key);
        });
        if (debug_rays && idx < 4) {
            #pragma omp critical
            {
                std::cout << "[debug] ray " << idx
                          << " o=(" << std::setprecision(6) << o.x << "," << o.y << "," << o.z << ")"
                          << " d=(" << d.x << "," << d.y << "," << d.z << ")\n";
                for (const auto& k : debug_hits) {
                    std::cout << "        voxel (" << k.x << "," << k.y << "," << k.z << ")\n";
                }
                std::cout << std::flush;
            }
        }
        py::ssize_t cur = ++done;
        if (done % progress_step == 0 || cur == total) {
            // 进度提示
            #pragma omp critical
            {
                std::cout << "Pre-rasterization progress: "
                        << static_cast<int>(100.0 * done / total) << "%\r" << std::flush;
            }
        }
    }
    std::unordered_set<VoxelKey, VoxelKeyHash> merged;
    size_t reserve_count = 0;
    for (const auto& s : local_sets) reserve_count += s.size();
    merged.reserve(reserve_count);
    for (const auto& s : local_sets) merged.insert(s.begin(), s.end());
#else
    std::unordered_set<VoxelKey, VoxelKeyHash> merged;
    merged.reserve(static_cast<size_t>(total));
    for (py::ssize_t idx = 0; idx < total; ++idx) {
        auto [o, d] = ray_accessor(idx);
        std::vector<VoxelKey> debug_hits;
        traverse_ray(o, d, grid, [&](const VoxelKey& key){
            merged.insert(key);
            if (debug_rays && idx < 4) debug_hits.push_back(key);
        });
        if (debug_rays && idx < 4) {
            std::cout << "[debug] ray " << idx
                      << " o=(" << std::setprecision(6) << o.x << "," << o.y << "," << o.z << ")"
                      << " d=(" << d.x << "," << d.y << "," << d.z << ")\n";
            for (const auto& k : debug_hits) {
                std::cout << "        voxel (" << k.x << "," << k.y << "," << k.z << ")\n";
            }
            std::cout << std::flush;
        }
    }
#endif

    // 将命中体素的顶点标记为 0
    for (const auto& key : merged) {
        auto it = grid.cells().find(key);
        if (it == grid.cells().end()) continue;
        for (auto& v : it->second.corners) {
            // if (!v->attr) v->attr = std::make_shared<IntAttribute>(1);
            auto ia = std::dynamic_pointer_cast<IntAttribute>(v->attr);
            if (ia) ia->value = 0;
        }
    }
}

VoxelGrid grad_sparsilization(const VoxelGrid& dense, int tolerant, bool invert) {
    VoxelGrid sparse;
    sparse.set_metadata(dense.origin(), dense.size(), dense.voxel_size(), dense.dims());

    auto cell_kept = [](const VoxelCell& cell) {
        for (const auto& v : cell.corners) {
            auto ia = std::dynamic_pointer_cast<IntAttribute>(v->attr);
            if (!ia || ia->value != 1) return false;
        }
        return true;
    };

    std::unordered_map<VoxelKey, bool, VoxelKeyHash> kept_map;
    kept_map.reserve(dense.cell_count());
    for (const auto& kv : dense.cells()) {
        kept_map.emplace(kv.first, cell_kept(kv.second));
    }

    // 构建邻域偏移
    std::vector<Vec3i> neighbor_offsets;
    if (tolerant == 1 || tolerant == 2) {
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    if (tolerant == 1 && (dx + dy + dz) != 1) continue; // 8 邻域（上下左右前后）
                    neighbor_offsets.emplace_back(dx, dy, dz);
                }
            }
        }
    }

    for (const auto& kv : dense.cells()) {
        const auto& cell = kv.second;
        bool keep = kept_map.at(cell.key);
        if (!keep && tolerant > 0) {
            for (const auto& off : neighbor_offsets) {
                VoxelKey nk{cell.key.x + off.x, cell.key.y + off.y, cell.key.z + off.z};
                auto it = kept_map.find(nk);
                if (it != kept_map.end() && it->second) { keep = true; break; }
            }
        }
        if (invert) keep = !keep;
        if (!keep) continue;
        VoxelCell* dst = sparse.ensure_cell(cell.key);
        // 顶点属性留空
        for (auto& v : dst->corners) v->attr.reset();
    }
    sparse.gc_vertices();
    return sparse;
}
