#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "pre_resterization.h"
#include "voxel_initialize.h"
#include "voxel_grid.h"
#include "voxel_node.h"
#include "rasterizer_torch.h"

namespace py = pybind11;

static VoxelKey key_from_tuple(const py::tuple& t) {
    if (t.size() != 3) throw std::invalid_argument("VoxelKey 需要 (x,y,z) 元组");
    return VoxelKey{t[0].cast<int64_t>(), t[1].cast<int64_t>(), t[2].cast<int64_t>()};
}

PYBIND11_MODULE(tetra_sh_shader_cpp, m) {
    m.doc() = "Sparse voxel grid with shared vertices and attribute handles.";

    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<double, double, double>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z);

    py::class_<Vec3i>(m, "Vec3i")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &Vec3i::x)
        .def_readwrite("y", &Vec3i::y)
        .def_readwrite("z", &Vec3i::z);

    py::class_<VoxelKey>(m, "VoxelKey")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &VoxelKey::x)
        .def_readwrite("y", &VoxelKey::y)
        .def_readwrite("z", &VoxelKey::z);

    py::class_<Attribute, std::shared_ptr<Attribute>>(m, "Attribute")
        .def(py::init<>());

    py::class_<IntAttribute, Attribute, std::shared_ptr<IntAttribute>>(m, "IntAttribute")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("value"))
        .def_readwrite("value", &IntAttribute::value);

    py::class_<TetraMesh>(m, "TetraMesh")
        .def(py::init<>())
        .def_readwrite("vertices", &TetraMesh::vertices)
        .def_readwrite("tets", &TetraMesh::tets)
        .def("write_ply", &TetraMesh::write_ply, py::arg("path"));

    py::class_<Vertex, std::shared_ptr<Vertex>>(m, "Vertex")
        .def_property_readonly("key", [](const Vertex& v) { return VoxelKey{v.key.x, v.key.y, v.key.z}; })
        .def_property("attr",
            [](const Vertex& v) { return v.attr; },
            [](Vertex& v, const std::shared_ptr<Attribute>& a) { v.attr = a; });

    py::class_<VoxelCell>(m, "VoxelCell")
        .def_property_readonly("key", [](const VoxelCell& c) { return VoxelKey{c.key.x, c.key.y, c.key.z}; })
        .def_property_readonly("corners", [](const VoxelCell& c) {
            std::vector<std::shared_ptr<Vertex>> out;
            out.reserve(8);
            for (const auto& v : c.corners) out.push_back(v);
            return out;
        })
        .def("to_tetra_mesh", [](const VoxelCell& c, const Vec3& origin, double voxel_size) {
            return c.to_tetra_mesh(origin, voxel_size);
        }, py::arg("origin"), py::arg("voxel_size"))
        .def("write_tetra_ply", &VoxelCell::write_tetra_ply,
             py::arg("origin"), py::arg("voxel_size"), py::arg("path"));

    py::class_<VoxelGrid>(m, "VoxelGrid")
        .def(py::init<>())
        .def("build_dense", &VoxelGrid::build_dense, py::arg("origin"), py::arg("size"), py::arg("voxel_size"))
        .def("ensure_cell", [](VoxelGrid& g, py::tuple key) { return g.ensure_cell(key_from_tuple(key)); },
             py::return_value_policy::reference_internal)
        .def("cell_count", &VoxelGrid::cell_count)
        .def("vertex_count", &VoxelGrid::vertex_count)
        .def("cell_corner_keys", [](const VoxelGrid& g, py::tuple key) { return g.cell_corner_keys(key_from_tuple(key)); })
        .def("gc_vertices", &VoxelGrid::gc_vertices)
        .def("cells", [](const VoxelGrid& g) {
            std::vector<VoxelCell> out;
            out.reserve(g.cells().size());
            for (const auto& kv : g.cells()) out.push_back(kv.second);
            return out;
        })
        .def("vertex_keys", [](const VoxelGrid& g) {
            std::vector<VoxelKey> out;
            out.reserve(g.vertex_count());
            for (const auto& kv : g.vertices()) out.push_back(kv.first);
            return out;
        })
        .def("to_tetra_mesh", &VoxelGrid::to_tetra_mesh)
        .def("write_tetra_ply", &VoxelGrid::write_tetra_ply, py::arg("path"));

    m.def("grid_init", &grid_init, "初始化所有顶点属性为1");
    m.def("image_pre_resterization", &image_pre_resterization,
          py::arg("grid"), py::arg("origins"), py::arg("dirs"),
          "沿射线遍历可达体素并将顶点属性置0");
    m.def("grad_sparsilization", &grad_sparsilization,
          py::arg("dense"), py::arg("tolerant") = 0, py::arg("invert") = false,
          "根据顶点属性（1保留/0剔除）生成稀疏网格，属性清空，可选容错邻域，invert 为反选开关");

    m.def("init_vertex_tensor", &init_vertex_tensor,
          py::arg("grid"), py::arg("feature_dim"),
          py::arg("on_cuda") = false, py::arg("fill_ones") = false, py::arg("as_int") = false,
          "创建 (N,C) torch 张量并将顶点 attr 指向对应行");
    m.def("attach_vertex_tensor", &attach_vertex_tensor,
          py::arg("grid"), py::arg("tensor"),
          "将已有的 (N,C) 张量挂接到顶点 attr（行序按 VoxelKey 排序）");
    m.def("ensure_active_cells_from_vertex_attr", &ensure_active_cells_from_vertex_attr,
          py::arg("grid"),
          "若体素八个角任一带 attr，则补齐该体素单元；返回新增数量");
    m.def("build_coarse_occupancy", &build_coarse_occupancy,
          py::arg("grid"), py::arg("res") = 8, py::arg("on_cuda") = false,
          "构建 coarse 占据网格 (res^3)，返回 uint8 torch 张量");
    m.def("build_coarse_index", &build_coarse_index,
          py::arg("grid"), py::arg("res") = 8, py::arg("on_cuda") = false,
          "构建 coarse 索引，返回 (occ uint8[res,res,res], offsets int64[B+1], keys int64[N,3], bitmask uint64[B])");
    m.def("rasterize_image", &rasterize_image,
          py::arg("grid"), py::arg("intrinsic"), py::arg("extrinsic"),
          py::arg("height"), py::arg("width"), py::arg("coarse_res") = 8,
          py::arg("vertex_features") = torch::Tensor(),
          "占位体素渲染：内部生成射线并调用 CUDA kernel，输出 HxWx3 颜色");
    m.def("rasterize_image_with_index", &rasterize_image_with_index,
          py::arg("grid"), py::arg("intrinsic"), py::arg("extrinsic"),
          py::arg("coarse_offsets"), py::arg("voxel_keys"), py::arg("coarse_mask"),
          py::arg("height"), py::arg("width"), py::arg("coarse_res") = 8,
          py::arg("vertex_features") = torch::Tensor(),
          "使用预计算 coarse 索引的占位渲染（避免每帧重建 coarse 索引），输出 HxWx3 颜色");
    m.def("rasterize_image_with_index_dense", &rasterize_image_with_index_dense,
          py::arg("grid"), py::arg("intrinsic"), py::arg("extrinsic"),
          py::arg("coarse_offsets"), py::arg("voxel_keys"), py::arg("coarse_mask"),
          py::arg("vertex_sigma"), py::arg("vertex_color"), py::arg("vertex_mask"),
          py::arg("height"), py::arg("width"), py::arg("coarse_res") = 8,
          "使用预计算 coarse 索引和稠密顶点网格的渲染（vertex_sigma/color/mask 已在外部构建），输出 HxWx3 颜色");
    m.def("build_dense_vertex_grids_from_features", &build_dense_vertex_grids_from_features,
          py::arg("grid"), py::arg("vertex_features"),
          "将顶点特征映射到稠密网格 (sigma, color, valid, xyz)");
}
