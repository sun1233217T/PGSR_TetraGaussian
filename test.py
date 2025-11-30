"""
Synthetic test pipeline and Colmap pipeline runner.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tetra_sh_shader import pre_resterization
from utils.graphics_utils import getWorld2View2


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
    """Simple pinhole camera looking at origin; keep only rays hitting the unit sphere."""
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

    oc = origins
    b = 2 * np.sum(oc * dirs, axis=1)
    c = np.sum(oc * oc, axis=1) - d
    disc = b * b - 4 * c
    mask = disc >= 0
    return origins[mask], dirs[mask]


def rays_from_image_mask(image_path, mask_path, intrinsic, extrinsic, resize=50):
    """
    Return positive (mask>0) and negative (mask==0) rays.
    """
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
        if img.shape[2] == 4:
            mask = img[:, :, 3]
            mask = (mask > 50).astype(np.uint8) * 255
        else:
            mask = np.ones((H, W), dtype=np.uint8) * 255
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    mask = np.logical_not(mask).astype(np.uint8)
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
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    fx /= resize
    fy /= resize
    cx /= resize
    cy /= resize

    def build_rays(xs, ys):
        if xs.size == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
        us = xs.astype(np.float64)
        vs = ys.astype(np.float64)
        dirs_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], axis=-1)
        dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1, keepdims=True)
        dirs_world = dirs_cam @ R.T
        origins = np.tile(t[None, :], (dirs_world.shape[0], 1))
        return origins, dirs_world

    origins_pos, dirs_world_pos = build_rays(xs_pos, ys_pos)
    origins_neg, dirs_world_neg = build_rays(xs_neg, ys_neg)
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
    parser.add_argument("--resize", type=int, default=50, help="rays_from_image_mask 的下采样倍数")
    parser.add_argument("--feature-dim", type=int, default=0, help=">0 时，对稀疏网格运行 vertex feature 初始化测试 (C 维)")
    parser.add_argument("--feature-device", type=str, default="cpu", help="feature 初始化的目标设备")
    parser.add_argument("--feature-fill", type=float, default=0.0, help="feature 初始化填充值（float 模式）")
    parser.add_argument("--feature-as-int", action="store_true", help="以 int64 而非 float 初始化 feature 张量")
    args = parser.parse_args()

    tsh = load_tetra()

    if args.cam_list:
        from scene.dataset_readers import sceneLoadTypeCallbacks
        scene = sceneLoadTypeCallbacks["Colmap"](Path(args.cam_list), "images", False)
        print(f"[test.py] loaded scene from {args.cam_list}")
        sparse = pre_resterization.pre_rasterize_scene(scene, voxel_size=args.voxel_size, resize=args.resize)
    elif args.image and args.intrinsic and args.extrinsic:
        o, d, o_neg, d_neg = rays_from_image_mask(args.image, args.mask, args.intrinsic, args.extrinsic, resize=args.resize)
        grid = tsh.VoxelGrid()
        origin_np = np.array([-args.size[0] / 2, -args.size[1] / 2, -args.size[2] / 2], dtype=np.float64)
        size_np = np.array(args.size, dtype=np.float64)
        origin_vec = tsh.Vec3(*origin_np.tolist())
        size_vec = tsh.Vec3(*size_np.tolist())
        grid.build_dense(origin_vec, size_vec, args.voxel_size)
        tsh.grid_init(grid)
        tsh.image_pre_resterization(grid, o_neg, d_neg)
        grid = tsh.grad_sparsilization(grid, tolerant=0, invert=True)
        tsh.grid_init(grid)
        tsh.image_pre_resterization(grid, o, d)
        sparse = tsh.grad_sparsilization(grid, tolerant=2, invert=False)
    else:
        o, d = generate_camera_rays(args.res[0], args.res[1], fov_deg=args.fov, cam_z=args.cam_z, d=args.ball_r)
        print(f"[test.py] generated {o.shape[0]} rays hitting the unit sphere")
        grid = tsh.VoxelGrid()
        origin_np = np.array([-args.size[0] / 2, -args.size[1] / 2, -args.size[2] / 2], dtype=np.float64)
        size_np = np.array(args.size, dtype=np.float64)
        origin_vec = tsh.Vec3(*origin_np.tolist())
        size_vec = tsh.Vec3(*size_np.tolist())
        grid.build_dense(origin_vec, size_vec, args.voxel_size)
        tsh.grid_init(grid)
        tsh.image_pre_resterization(grid, o, d)
        sparse = tsh.grad_sparsilization(grid, tolerant=2, invert=False)

    out_path = Path(args.output)
    sparse.write_tetra_ply(str(out_path))
    print(f"[test.py] wrote {out_path}")

    if args.feature_dim > 0:
        feats = tsh.initialize_vertex_features(
            sparse,
            args.feature_dim,
            device=args.feature_device,
            fill_value=args.feature_fill,
            as_int=args.feature_as_int,
        )
        print(f"[test.py] initialized vertex features: shape={tuple(feats.shape)}, device={feats.device}, dtype={feats.dtype}")
        if feats.numel() > 0:
            sample = feats[0].detach().cpu().numpy()
            print(f"[test.py] sample[0]: {sample}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
