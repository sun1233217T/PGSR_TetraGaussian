"""
Synthetic test pipeline:
1) 构建致密体素网格
2) 生成命中单位球的多视角射线
3) grid_init -> image_pre_resterization -> grad_sparsilization
4) 导出稀疏体素四面体为 PLY
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from utils.graphics_utils import getWorld2View2
from mtools import debug

def load_tetra():
    try:
        import tetra_sh_shader as tsh  # type: ignore
        return tsh
    except Exception:
        repo_root = Path(__file__).resolve().parent
        pkg_root = repo_root / "submodules" / "tetra_sh_shader" / "python"
        sys.path.insert(0, str(pkg_root))
        import importlib
        import tetra_sh_shader as tsh  # type: ignore
        return importlib.reload(tsh)


def generate_camera_rays(W, H, fov_deg=60.0, cam_z=-2.0, d=1.0):
    """
    简单 pinhole 相机，光心(0,0,cam_z)，朝 +z，看向原点。
    返回 origins (K,3), dirs (K,3) ，只保留命中单位球的射线。
    """
    fov = np.deg2rad(fov_deg)
    aspect = W / H
    xs = (np.arange(W) + 0.5) / W * 2 - 1
    ys = (np.arange(H) + 0.5) / H * 2 - 1
    xs = xs * np.tan(fov / 2) * aspect
    ys = ys * np.tan(fov / 2)
    xv, yv = np.meshgrid(xs, ys)
    dirs = np.stack([xv, -yv, np.ones_like(xv)], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    origin = np.array([0.0, 0.0, cam_z], dtype=np.float64)
    origins = np.repeat(origin[None, :], W * H, axis=0)
    dirs = dirs.reshape(-1, 3)

    # 判定是否与单位球相交：t^2*d.d + 2t o.d + (o.o - 1) = 0
    oc = origins
    b = 2 * np.sum(oc * dirs, axis=1)
    c = np.sum(oc * oc, axis=1) - d
    disc = b * b - 4 * c
    mask = disc >= 0
    return origins[mask], dirs[mask]

def draw_rays(o,d):
    import plyfile
    points = []
    for i in range(o.shape[0]):
        for t in range(1000):
            points.append(o[i] + d[i,0] * (t * 0.01))
    points = np.array(points)
    vertex = np.array([tuple(p) for p in points],   
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex, 'vertex')])
    ply.write('debug_rays.ply')


def rays_from_image_mask(image_path, mask_path, intrinsic, extrinsic, resize = 50):
    import cv2

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取 image: {image_path}")
    H, W = img.shape[:2]
    H //= resize
    W //= resize
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取 mask: {mask_path}")
        if mask.shape[:2] != (H, W):
            raise ValueError("image 与 mask 尺寸不一致")
    else:
        # 默认用 alpha 通道；若不存在则全 1
        if img.shape[2] == 4:
            mask = img[:, :, 3]
            mask = (mask > 50).astype(np.uint8) * 255
        else:
            mask = np.ones((H, W), dtype=np.uint8) * 255
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    mask = np.logical_not(mask).astype(np.uint8)
    # debug()
    cv2.imwrite("debug_mask.png", mask * 255)

    def to_mat(x):
        if isinstance(x, str):
            if x.endswith(".npy"):
                return np.load(x)
            return np.loadtxt(x)
        return np.asarray(x)

    K = to_mat(intrinsic).reshape(3, 3)
    ext = to_mat(extrinsic).reshape(4, 4)  # world->cam
    cam2world = np.linalg.inv(ext)
    R = cam2world[:3, :3]
    t = cam2world[:3, 3]

    ys_pos, xs_pos = np.nonzero(mask > 0)
    ys_neg, xs_neg = np.nonzero(mask == 0)
    us = xs_pos.astype(np.float64)
    vs = ys_pos.astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    fx /= resize
    fy /= resize
    cx /= resize
    cy /= resize
    dirs_cam_pos = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], axis=-1)
    dirs_cam_pos = dirs_cam_pos / np.linalg.norm(dirs_cam_pos, axis=1, keepdims=True)
    dirs_world_pos = dirs_cam_pos @ R.T
    origins_pos = np.tile(t[None, :], (dirs_world_pos.shape[0], 1))

    us_neg = xs_neg.astype(np.float64)
    vs_neg = ys_neg.astype(np.float64)
    dirs_cam_neg = np.stack([(us_neg - cx) / fx, (vs_neg - cy) / fy, np.ones_like(us_neg)], axis=-1)
    dirs_cam_neg = dirs_cam_neg / np.linalg.norm(dirs_cam_neg, axis=1, keepdims=True)
    dirs_world_neg = dirs_cam_neg @ R.T
    origins_neg = np.tile(t[None, :], (dirs_world_neg.shape[0], 1))

    return origins_pos, dirs_world_pos, origins_neg, dirs_world_neg


def load_cameras_from_file(path: Path):
    from scene.dataset_readers import sceneLoadTypeCallbacks
    scene = sceneLoadTypeCallbacks["Colmap"](path, "images", False)
    cam_infos = list(scene.train_cameras) + list(scene.test_cameras)

    cams = []
    for cam in cam_infos:
        W, H = cam.width, cam.height
        fx, fy = cam.fx, cam.fy
        cx, cy = W * 0.5, H * 0.5
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        w2c = getWorld2View2(cam.R, cam.T)
        cams.append({
            "image": str(cam.image_path),
            "mask": None,  # 使用 alpha
            "intrinsic": K,
            "extrinsic": w2c,
        })

    bbox_origin, bbox_size = None, None
    if scene.point_cloud is not None and scene.point_cloud.points is not None:
        pts = np.asarray(scene.point_cloud.points)
        if pts.size > 0:
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            diag = np.linalg.norm(mx - mn)
            pad = 0.05 * diag
            bbox_origin = mn - pad
            bbox_size = (mx - mn) + 2 * pad
    return cams, bbox_origin, bbox_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel-size", type=float, default=0.05, help="体素边长")
    parser.add_argument("--size", type=float, nargs=3, default=[1.5, 1.5, 1.5], help="构建尺寸 (x y z)")
    parser.add_argument("--output", type=str, default="sparse_grid.ply", help="稀疏体素导出的 PLY 路径")
    parser.add_argument("--res", type=int, nargs=2, default=[64, 64], help="相机分辨率 W H（合成模式）")
    parser.add_argument("--fov", type=float, default=60.0, help="相机视场角(度，合成模式)")
    parser.add_argument("--cam-z", type=float, default=-2.0, help="相机 z 位置（合成模式）")
    parser.add_argument("--ball_r", type=float, default=1.0, help="中心球半径（合成模式）")
    parser.add_argument("--image", type=str, default=None, help="输入图像路径（单张）")
    parser.add_argument("--mask", type=str, default=None, help="输入不透明掩码路径（单张）")
    parser.add_argument("--intrinsic", type=str, default=None, help="相机内参 (3x3 npy/txt，单张)")
    parser.add_argument("--extrinsic", type=str, default=None, help="相机外参 world->cam (4x4 npy/txt，单张)")
    parser.add_argument("--cam-list", type=str, default=None,
                        help="相机列表文件（image [mask] intrinsic extrinsic）或 Colmap 数据集路径（自动用 alpha 掩码）")
    args = parser.parse_args()

    tsh = load_tetra()

    cams = []
    bbox_origin = bbox_size = None
    if args.cam_list:
        cams, bbox_origin, bbox_size = load_cameras_from_file(Path(args.cam_list))
    if cams:
        all_o, all_d = [], []
        all_o_neg, all_d_neg = [], []
        for cam in cams:
            o, d, o_neg, d_neg = rays_from_image_mask(cam["image"], cam["mask"], cam["intrinsic"], cam["extrinsic"])
            all_o.append(o); all_d.append(d.reshape(-1, 1, 3))
            all_o_neg.append(o_neg); all_d_neg.append(d_neg.reshape(-1, 1, 3))
            # if (len(all_o)) == 1:
            #     break  # DEBUG: 只用第一张
        o = np.concatenate(all_o, axis=0)
        d = np.concatenate(all_d, axis=0)
        o_neg = np.concatenate(all_o_neg, axis=0)
        d_neg = np.concatenate(all_d_neg, axis=0)
        print(f"[test.py] loaded {o.shape[0]} rays from cam list ({len(cams)} cams)")
    elif args.image and args.intrinsic and args.extrinsic:
        o, d, o_neg, d_neg = rays_from_image_mask(args.image, args.mask, args.intrinsic, args.extrinsic)
        print(f"[test.py] loaded {o.shape[0]} rays from single image (mask {('alpha' if args.mask is None else args.mask)})")
    else:
        o, d = generate_camera_rays(args.res[0], args.res[1], fov_deg=args.fov, cam_z=args.cam_z, d=args.ball_r)
        print(f"[test.py] generated {o.shape[0]} rays hitting the unit sphere")

    import time
    t0 = time.time()
    grid = tsh.VoxelGrid()
    if bbox_origin is not None and bbox_size is not None:
        origin_np = bbox_origin
        size_np = bbox_size
    else:
        origin_np = np.array([-args.size[0] / 2, -args.size[1] / 2, -args.size[2] / 2], dtype=np.float64)
        size_np = np.array(args.size, dtype=np.float64)
    origin_vec = tsh.Vec3(*origin_np.tolist())
    size_vec = tsh.Vec3(*size_np.tolist())
    grid.build_dense(origin_vec, size_vec, args.voxel_size)
    t_build = time.time()

    tsh.grid_init(grid)
    t_init = time.time()
    print(f"[test.py] built dense grid: cells={grid.cell_count()}, vertices={grid.vertex_count()}")

    t_rays = time.time()
    tsh.image_pre_resterization(grid, o_neg, d_neg)
    grid = tsh.grad_sparsilization(grid, tolerant = 0, invert = True)
    tsh.grid_init(grid)
    
    t_mask = time.time()

    # draw_rays(o, d)

    tsh.image_pre_resterization(grid, o, d)
    sparse = tsh.grad_sparsilization(grid, tolerant = 2, invert = False)
    t_sparse = time.time()
    print(f"[test.py] dense cells={grid.cell_count()}, sparse cells={sparse.cell_count()}, vertices={sparse.vertex_count()}")

    out_path = Path(args.output)
    sparse.write_tetra_ply(str(out_path))
    t_ply = time.time()
    print(f"[test.py] wrote {out_path}")
    print(f"[timing] build_dense: {(t_build - t0)*1000:.2f} ms")
    print(f"[timing] grid_init:   {(t_init - t_build)*1000:.2f} ms")
    print(f"[timing] rays:        {(t_rays - t_init)*1000:.2f} ms")
    print(f"[timing] neg_mask pass:   {(t_mask - t_rays)*1000:.2f} ms")
    print(f"[timing] mask pass:    {(t_sparse - t_mask)*1000:.2f} ms")
    print(f"[timing] write ply:   {(t_ply - t_sparse)*1000:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
