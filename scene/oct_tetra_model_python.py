import numpy as np
from mtools import load_gs, debug
from collections import deque
import open3d as o3d
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

def gather_nodes(root, leaves_only=True, level_range=None, max_nodes=None):
    """收集需要可视化的节点。"""
    out = []
    if root is None:
        return out
    dq = deque([root])
    while dq:
        n = dq.popleft()
        if getattr(n, "is_dead", False):
            continue
        ok_level = True
        if level_range is not None:
            lo, hi = level_range
            ok_level = (lo <= n.level <= hi)
        take = (n.is_leaf if leaves_only else True) and ok_level
        if take:
            out.append(n)
            if (max_nodes is not None) and (len(out) >= max_nodes):
                break
        if not n.is_leaf:
            dq.extend(n.children)
    return out

def build_lines_for_boxes(nodes, color_by_level=True):
    """
    把一批节点拼成一个 LineSet（每个节点一个线框立方体，共 12 根边）。
    """
    # 立方体 8 个顶点（单位立方，原点在最小角），和 12 条边的索引
    unit_pts = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],  # z=0
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],  # z=1
    ], dtype=np.float64)
    unit_edges = np.array([
        [0,1],[1,2],[2,3],[3,0],  # bottom
        [4,5],[5,6],[6,7],[7,4],  # top
        [0,4],[1,5],[2,6],[3,7],  # pillars
    ], dtype=np.int32)

    points = []
    lines = []
    colors = []

    # 统计层级做上色
    lv_min = min(n.level for n in nodes) if nodes else 0
    lv_max = max(n.level for n in nodes) if nodes else 1
    lv_span = max(1, lv_max - lv_min)

    base = 0
    for n in nodes:
        hs = float(n.half_size)
        c = np.asarray(n.center, dtype=np.float64)
        # 将单位立方缩放到 2*hs，再平移到 (center - hs)
        pts = unit_pts * (2.0 * hs) + (c - hs)
        points.append(pts)

        lines.append(unit_edges + base)
        base += 8

        if color_by_level:
            t = (n.level - lv_min) / lv_span  # 0~1
            # 简单渐变：蓝(低层) -> 红(高层)
            col = [t, 0.2, 1.0 - t]
        else:
            col = [0.9, 0.6, 0.2]
        colors.append(np.repeat([col], 12, axis=0))  # 每条边都上同色

    if not points:
        return o3d.geometry.LineSet()

    points = np.vstack(points)
    lines = np.vstack(lines)
    colors = np.vstack(colors)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def visualize_octree_lines(octree, leaves_only=True, level_range=None, max_nodes=200000,
                           add_points=None):
    """
    可视化：将八叉树节点画成线框立方体。
    - leaves_only: 仅叶子；False 则包含所有层节点
    - level_range: (lo, hi) 过滤层级
    - max_nodes:   最多渲染多少个节点（避免过重）
    - add_points:  可附带点云 (numpy (N,3)) 一起显示
    """
    nodes = gather_nodes(octree.root, leaves_only=leaves_only,
                         level_range=level_range, max_nodes=max_nodes)
    ls = build_lines_for_boxes(nodes, color_by_level=True)

    geoms = [ls]
    if add_points is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(add_points.astype(np.float64))
        geoms.append(pcd)

    o3d.visualization.draw_geometries(geoms, window_name="Octree (wire boxes)")

class Node:
    def __init__(self, half_size, center, is_leaf=True, level=0, father=None, para=None):

        self.half_size = half_size
        self.center = center
        self.is_leaf = is_leaf
        self.level = level
        self.father = father
        self.children = []
        self.is_dead = False
        self.para = para
        self.point_indices = []       # 仅叶子保有点索引（split 后清空父节点）

    def split(self):
        if not self.is_leaf:
            return
        hs_child = self.half_size * 0.5
        # 八个象限的偏移（xyz 分别取 -1 或 +1）
        offsets = np.array([
            [-1, -1, -1],
            [ 1, -1, -1],
            [-1,  1, -1],
            [ 1,  1, -1],
            [-1, -1,  1],
            [ 1, -1,  1],
            [-1,  1,  1],
            [ 1,  1,  1],
        ], dtype=np.float64) * hs_child

        for i in range(8):
            c = self.center + offsets[i]
            self.children.append(Node(hs_child, c, is_leaf=True, level=self.level+1, father=self))

        self.is_leaf = False

    def get_leaf_nodes(self):
        if self.is_leaf:
            return [self]
        else:
            leaves = []
            for child in self.children:
                leaves.extend(child.get_leaf_nodes())
            return leaves
        
    def is_dead(self):
        return self.is_dead
    
    def kill(self):
        self.is_dead = True

    def collect_leaf_nodes(self,output_list):
        if self.is_dead:
            return
        if self.is_leaf:
            output_list.append(self)
        else:
            for child in self.children:
                child.collect_leaf_nodes(output_list)

class Octree:
    def __init__(self):
        self.root = None

    def build(self):
        nodes_to_split = [self.root]
        while nodes_to_split:
            current_node = nodes_to_split.pop()
            if current_node.level < self.max_level:
                current_node.split()
                nodes_to_split.extend(current_node.children)
    
    def collect_leaf_nodes(self):
        output_list = []
        self.root.collect_leaf_nodes(output_list)
        return output_list
    
    @staticmethod
    def _to_numpy(points_obj):
        """将输入安全转成 float64 的 ndarray，形状 (N,3)。"""
        if _HAS_TORCH and isinstance(points_obj, torch.Tensor):
            arr = points_obj.detach().cpu().numpy()
        else:
            arr = np.asarray(points_obj)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"points should have shape (N,3), got {arr.shape}")
        return arr.astype(np.float64, copy=False)

    def build_from_points(self, points_data, c=1.0, max_depth=16, min_half_size=1e-9, up_level_c=0):
        """
        由点云构建八叉树（与给定 C++ 版本一致）：
          - 分裂阈值：threshold(level) = c + 0.5 * level
          - BFS 分裂；超过 max_depth 或 half_size 太小就不再分裂
        """
        pts = self._to_numpy(points_data)
        N = pts.shape[0]

        if N == 0:
            # 空点云：清空为一个“死节点”
            self.root = None
            return

        # 计算包围盒
        mn = np.min(pts, axis=0)  # (3,)
        mx = np.max(pts, axis=0)  # (3,)
        center = (mn + mx) * 0.5
        side = float(np.max(mx - mn))
        if side <= 0.0:
            side = 1e-9  # 所有点重合时避免 0 尺寸

        # 重建 root，并把所有点先放入 root
        self.root = Node(half_size=side * 0.5, center=center, is_leaf=True, level=0, father=None)
        self.root.point_indices = list(range(N))

        # 阈值函数
        def threshold(level: int) -> float:
            return float(c) + up_level_c * float(level)

        # BFS 队列
        q = deque([self.root])

        while q:
            node = q.popleft()
            cnt = len(node.point_indices)
            thr = threshold(node.level)

            # 分裂条件：点数 > 阈值 且 未超过 max_depth 且 尺寸仍可分
            if (cnt > thr) and (node.level < max_depth) and (node.half_size > min_half_size):
                # 分裂
                node.split()

                # 把当前节点的点分配到 8 个子节点（与 C++ 的 bit 规则一致）
                cx, cy, cz = node.center
                child_lists = [[] for _ in range(8)]
                for idx in node.point_indices:
                    x, y, z = pts[idx]
                    code = 0
                    if x >= cx: code |= 1   # bit 0: x+
                    if y >= cy: code |= 2   # bit 1: y+
                    if z >= cz: code |= 4   # bit 2: z+
                    child_lists[code].append(idx)

                # 写回子节点
                for i in range(8):
                    child = node.children[i]
                    child.point_indices = child_lists[i]
                    q.append(child)

                # 清空父节点的索引
                node.point_indices = []
            # 否则不再分裂（保持叶子 & 保留索引）

            #清空所有没有点的节点
            if len(node.point_indices) == 0 and node.is_leaf:
                node.kill()



def test():
    # 测试：从文件加载点云，构建八叉树
    xyz, opacities, scales, rots, features_dc, features_extra = load_gs("E:/tmp/zhutest1/point_cloud/iteration_7000/point_cloud_ori.ply",return_feature=True)
    # debug()
    octree = Octree()
    octree.build_from_points(xyz, c=1.0, max_depth=16, min_half_size=1e-9, up_level_c=0.25)

    # 收集叶子节点
    leaves = octree.collect_leaf_nodes()
    print(f"Total leaf nodes: {len(leaves)}")
    max_level = max(leaf.level for leaf in leaves) if leaves else 0
    print(f"Max leaf level: {max_level}")
    # for i, leaf in enumerate(leaves):
    #     if len(leaf.point_indices) > 1:
    #         print(f"Leaf {i}: Level {leaf.level}, Center {leaf.center}, Half-size {leaf.half_size}, Points {len(leaf.point_indices)}")
    #         for idx in leaf.point_indices:
    #             print(f"  Point index: {idx}, Coordinates: {xyz[idx]}")

    # 可视化八叉树线框
    leaves_only = True   # 若想看所有层的“子节点”，可设为 False
    level_range = None   # 例如 (5, 10) 只看 5~10 层
    visualize_octree_lines(octree, leaves_only=leaves_only,
                        level_range=level_range,
                        max_nodes=300000,
                        add_points=xyz)   # xyz 可选：把原点云一起显示


if __name__ == "__main__":
    test()