#ifndef OCTREE_PYBIND_HPP
#define OCTREE_PYBIND_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <memory>
#include <queue>
#include <stdexcept>
#include <limits>
#include <numeric>

namespace py = pybind11;

// 三维向量
struct Vec3 {
    double x{}, y{}, z{};
    Vec3() = default;
    Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
};

// 节点
class OctreeNode {
public:
    Vec3 center;               // 节点中心
    double halfSize;           // 半边长
    OctreeNode* children[8];
    OctreeNode* father;        // 父节点（非拥有）
    int level{0};              // 层级：root=0
    std::vector<size_t> point_indices; // 落在该节点内的点索引（叶上最有用）
    bool is_dead_flag{false}; // 标记为“死”节点（无效）
    // OctreeNode* eight_neighbors[8]{nullptr}; // 八邻居（非拥有）
    std::vector<double> para; // 存储参数

    OctreeNode(const Vec3& c, double hs, OctreeNode* f = nullptr, int lvl = 0)
        : center(c), halfSize(hs), father(f), level(lvl) {
        for (int i = 0; i < 8; ++i) children[i] = nullptr;
    }

    ~OctreeNode() { merge(); }

    bool is_leaf() const { return children[0] == nullptr; }

    bool is_last_father() const {
        if (is_leaf()) return false;
        for (int i = 0; i < 8; ++i)
            if (!children[i]->is_leaf()) return false;
        return true;
    }

    bool is_dead() const { return is_dead_flag; }

    void kill() { is_dead_flag = true; }

    void relive() { is_dead_flag = false; }

    void split() {
        if (!is_leaf()) return;
        double hs = halfSize * 0.5;
        for (int i = 0; i < 8; ++i) {
            Vec3 offset = {
                (i & 1) ? hs : -hs,
                (i & 2) ? hs : -hs,
                (i & 4) ? hs : -hs
            };
            Vec3 childCenter = { center.x + offset.x,
                                 center.y + offset.y,
                                 center.z + offset.z };
            children[i] = new OctreeNode(childCenter, hs, this, level + 1);
            children[i]->para = this->para; // 继承父节点参数
        }
    }

    void merge() {
        if (is_leaf()) return;
        for (int i = 0; i < 8; ++i) {
            if (!children[i]->is_leaf()) children[i]->merge();
            delete children[i];
            children[i] = nullptr;
        }
    }

    void collect_leaf_nodes(std::vector<OctreeNode*>& out) {
        if (is_dead()) return;
        if (is_leaf()) {
            out.push_back(this);
        } else {
            for (int i = 0; i < 8; ++i) children[i]->collect_leaf_nodes(out);
        }
    }

    void collect_last_father_nodes(std::vector<OctreeNode*>& out) {
        if (is_dead()) return;
        if (is_last_father()) {
            out.push_back(this);
        } else if (!is_leaf()) {
            for (int i = 0; i < 8; ++i) children[i]->collect_last_father_nodes(out);
        }
    }

    OctreeNode* get_child(int index) {
        if (index < 0 || index >= 8) return nullptr;
        return children[index];
    }

    std::vector<OctreeNode*> get_childrens() {
        std::vector<OctreeNode*> childNodes;
        for (int i = 0; i < 8; ++i) if (children[i]) childNodes.push_back(children[i]);
        return childNodes;
    }

    const Vec3& get_center() const { return center; }
    double get_half_size() const { return halfSize; }
    OctreeNode* get_father() const { return father; }
    int get_level() const { return level; }
    size_t num_points() const { return point_indices.size(); }
    std::vector<size_t> get_point_indices_copy() const { return point_indices; }
};

// 八叉树
class Octree {
public:
    OctreeNode* root;

    Octree(const Vec3& center, double size)
        : root(new OctreeNode(center, size * 0.5, nullptr, 0)) {}

    ~Octree() { delete root; }

    void split_root() { if (root) root->split(); }
    void merge_root() { if (root) root->merge(); }

    std::vector<OctreeNode*> get_all_leaf_nodes() {
        std::vector<OctreeNode*> leaves;
        if (root) root->collect_leaf_nodes(leaves);
        return leaves;
    }

    void store_para_to_oct_1d(std::vector<double>& para){
        std::vector<OctreeNode*> leaves;
        root->collect_leaf_nodes(leaves);
        if (leaves.size() != para.size()){
            throw std::invalid_argument("para size not match leaf node size");
        }
        for (size_t i = 0; i < leaves.size(); i++){
            leaves[i]->para.push_back(para[i]);
        }
    }

    std::vector<double> get_para_from_oct_1d(){
        std::vector<OctreeNode*> leaves;
        root->collect_leaf_nodes(leaves);
        std::vector<double> para;
        para.reserve(leaves.size());
        for (size_t i = 0; i < leaves.size(); i++){
            if (leaves[i]->para.size() == 0){
                throw std::invalid_argument("leaf node has no para");
            }
            para.push_back(leaves[i]->para[0]);
        }
        return para;
    }

    std::vector<double> get_all_leaf_xyzhw(){
        std::vector<OctreeNode*> leaves;
        root->collect_leaf_nodes(leaves);
        std::vector<double> xyzhw;
        xyzhw.reserve(leaves.size() * 4);
        for (size_t i = 0; i < leaves.size(); i++){
            xyzhw.push_back(leaves[i]->center.x);
            xyzhw.push_back(leaves[i]->center.y);
            xyzhw.push_back(leaves[i]->center.z);
            xyzhw.push_back(leaves[i]->halfSize);
        }
        return xyzhw;
    }

    std::vector<OctreeNode*> get_all_last_father_nodes() {
        std::vector<OctreeNode*> fathers;
        if (root) root->collect_last_father_nodes(fathers);
        return fathers;
    }

    void relive_dead_ndoes() {
        if (!root) return;
        std::queue<OctreeNode*> q; q.push(root);
        while (!q.empty()) {
            OctreeNode* node = q.front(); q.pop();
            if (node->is_dead()) node->relive();
            if (!node->is_leaf()) {
                for (int i = 0; i < 8; ++i) q.push(node->children[i]);
            }
        }
    }

    // 由点云初始化：points 可为 numpy.ndarray (N,3) 或 torch.Tensor(N,3)
    void build_from_points(py::object points_obj, double c = 0.0,
                           int max_depth = 16, double min_half_size = 1e-9) {
        // 1) 将输入安全地转成 float64 的 ndarray
        std::vector<Vec3> pts = to_vec3(points_obj);

        if (pts.empty()) {
            // 空点云：清空为一个“死”节点
            if (root) { delete root; root = nullptr; }
            return;
        }

        // 2) 计算立方包围盒并重建 root
        Vec3 mn{+std::numeric_limits<double>::infinity(),
                +std::numeric_limits<double>::infinity(),
                +std::numeric_limits<double>::infinity()};
        Vec3 mx{-std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity()};
        for (const auto& p : pts) {
            mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z);
            mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y); mx.z = std::max(mx.z, p.z);
        }
        Vec3 ccenter{ (mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, (mn.z + mx.z) * 0.5 };
        double side = std::max({ mx.x - mn.x, mx.y - mn.y, mx.z - mn.z });
        if (side <= 0) side = 1e-9; // 所有点重合时避免 0 尺寸
        if (root) delete root;
        root = new OctreeNode(ccenter, side * 0.5, nullptr, 0);

        // 3) 将所有点先放入 root
        root->point_indices.resize(pts.size());
        std::iota(root->point_indices.begin(), root->point_indices.end(), 0);

        // 4) BFS 分裂：阈值 = c + 0.5 * level
        auto threshold = [&](int level) -> double { return c + 0.5 * double(level); };
        std::queue<OctreeNode*> q; q.push(root);

        while (!q.empty()) {
            OctreeNode* node = q.front(); q.pop();
            const size_t cnt = node->point_indices.size();
            const double thr = threshold(node->level);

            // 分裂条件：点数 > 阈值 且 未超过 max_depth 且 尺寸仍可分
            if (cnt > thr && node->level < max_depth && node->halfSize > min_half_size) {
                node->split();

                // 把当前节点的点分配到 8 个子节点
                for (size_t idx : node->point_indices) {
                    const Vec3& p = pts[idx];
                    int code = 0;
                    if (p.x >= node->center.x) code |= 1; // x 正半轴 -> bit 0
                    if (p.y >= node->center.y) code |= 2; // y 正半轴 -> bit 1
                    if (p.z >= node->center.z) code |= 4; // z 正半轴 -> bit 2
                    node->children[code]->point_indices.push_back(idx);
                }
                node->point_indices.clear(); // 释放父节点存的索引

                for (int i = 0; i < 8; ++i) q.push(node->children[i]);
            }
        }
    }

private:
    // 将 Python 对象转为 std::vector<Vec3>；支持：
    // - numpy.ndarray (N,3)
    // - torch.Tensor (N,3)（CPU/GPU；GPU会自动 .detach().cpu().numpy()）
    static std::vector<Vec3> to_vec3(const py::object& obj) {
        py::module np = py::module::import("numpy");
        py::object np_float64 = np.attr("float64");

        py::array arr;

        // 先尝试直接 asarray（CPU numpy 或 CPU torch.Tensor 有时可行）
        bool ok = true;
        try {
            arr = np.attr("asarray")(obj, py::arg("dtype") = np_float64).cast<py::array>();
        } catch (const py::error_already_set&) {
            ok = false;
        }

        // 如果失败，尝试 torch 路径：detach().cpu().numpy()
        if (!ok) {
            try {
                py::object torch = py::module::import("torch");
                py::object as_numpy =
                    np.attr("asarray")(obj.attr("detach")().attr("cpu")().attr("numpy")(),
                                       py::arg("dtype") = np_float64);
                arr = as_numpy.cast<py::array>();
                ok = true;
            } catch (const std::exception&) {
                // 留给下面的错误统一抛
            }
        }

        if (!ok) {
            throw std::invalid_argument(
                "points 必须是形状 (N,3) 的 numpy.ndarray 或可转换的 torch.Tensor（若在 GPU 将自动转 CPU）");
        }

        py::buffer_info info = arr.request();
        if (info.ndim != 2 || info.shape[1] != 3) {
            throw std::invalid_argument("points 形状必须为 (N,3)");
        }
        const py::ssize_t N = info.shape[0];
        const double* data = static_cast<const double*>(info.ptr);

        // 假设行主、可通过步长乘法访问（即使非连续也可用步长）
        std::vector<Vec3> out;
        out.reserve(static_cast<size_t>(N));

        const auto stride0 = info.strides[0] / static_cast<py::ssize_t>(sizeof(double));
        const auto stride1 = info.strides[1] / static_cast<py::ssize_t>(sizeof(double));

        for (py::ssize_t i = 0; i < N; ++i) {
            const double* row = data + i * stride0;
            out.emplace_back(row[0 * stride1], row[1 * stride1], row[2 * stride1]);
        }
        return out;
    }
};

// 绑定
PYBIND11_MODULE(octree, m) {
    m.doc() = "Octree spatial data structure (point-cloud aware)";

    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<double, double, double>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z);

    py::class_<OctreeNode, std::unique_ptr<OctreeNode, py::nodelete>>(m, "OctreeNode")
        .def_property_readonly("center", [](OctreeNode &n) { return n.center; })
        .def_property_readonly("half_size", [](OctreeNode &n) { return n.halfSize; })
        .def_property_readonly("level", &OctreeNode::get_level, "Depth level of this node (root=0).")
        .def("is_leaf", &OctreeNode::is_leaf)
        .def("split", &OctreeNode::split)
        .def("merge", &OctreeNode::merge)
        .def("num_points", &OctreeNode::num_points, "Number of points in this node (for leaves).")
        .def("get_point_indices", &OctreeNode::get_point_indices_copy,
             "Return a copy of point indices contained in this node.")
        .def("get_child", &OctreeNode::get_child, py::arg("index"),
             "Get child node by index (0-7). Returns nullptr if index is out of range.")
        .def("get_childrens", &OctreeNode::get_childrens,
             "Get all child nodes (non-owning pointers).")
        .def("get_father", &OctreeNode::get_father,
             py::return_value_policy::reference_internal,
             "Get the parent node (non-owning pointer).")
        .def("get_center", &OctreeNode::get_center,
             py::return_value_policy::reference_internal,
             "Get the center of the node (Vec3).")
        .def("get_half_size", &OctreeNode::get_half_size,
             "Get the half size of the node (double).")
        .def("collect_leaf_nodes", &OctreeNode::collect_leaf_nodes, py::arg("out"))
        .def("is_dead", &OctreeNode::is_dead)
        .def("kill", &OctreeNode::kill);

    py::class_<Octree>(m, "Octree")
        .def(py::init<const Vec3&, double>(), py::arg("center"), py::arg("size"))
        .def("split_root", &Octree::split_root)
        .def("merge_root", &Octree::merge_root)
        .def("get_all_leaf_nodes", &Octree::get_all_leaf_nodes,
             py::return_value_policy::reference_internal)
        .def("get_all_last_father_nodes", &Octree::get_all_last_father_nodes,
             py::return_value_policy::reference_internal)
        .def("build_from_points", &Octree::build_from_points,
             py::arg("points"), py::arg("c"),
             py::arg("max_depth") = 32, py::arg("min_half_size") = 1e-9,
             "Initialize and split the tree from a point cloud. "
             "points: numpy (N,3) or torch.Tensor (N,3). "
             "Split rule: if count > c + 0.5*level, split until constraints.")
        .def("store_para_to_oct_1d", &Octree::store_para_to_oct_1d, py::arg("para"),
             "Store a list of parameters (one per leaf node) into the octree leaves.")
        .def("get_para_from_oct_1d", &Octree::get_para_from_oct_1d,
             "Retrieve a list of parameters (one per leaf node) from the octree leaves.")
        .def("get_all_leaf_xyzhw", &Octree::get_all_leaf_xyzhw,
             "Get a flat list of [x,y,z,half_size] for all leaf nodes.");
}

#endif // OCTREE_PYBIND_HPP


// c++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) -I${CONDA_PREFIX}/include octgs_model.cpp -o octree$(python3-config --extension-suffix) 