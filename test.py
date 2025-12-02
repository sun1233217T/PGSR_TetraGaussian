"""
Synthetic test pipeline and Colmap pipeline runner.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tetra_sh_shader import pre_resterization
from tetra_sh_shader import rasterize, rasterize_with_coarse, build_coarse_index
from utils.graphics_utils import getWorld2View2
import cv2
import time

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
    mask = disc <= 0
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
    parser.add_argument("--feature-device", type=str, default="cuda", help="feature 初始化的目标设备")
    parser.add_argument("--feature-fill", type=float, default=0.0, help="feature 初始化填充值（float 模式）")
    parser.add_argument("--feature-as-int", action="store_true", help="以 int64 而非 float 初始化 feature 张量")
    parser.add_argument("--dump-coarse", action="store_true", help="构建 coarse 占据网格并打印统计")
    parser.add_argument("--coarse-res", type=int, default=8, help="coarse 网格边长")
    parser.add_argument("--coarse-device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="coarse 网格输出设备")
    parser.add_argument("--dump-coarse-index", action="store_true", help="构建 coarse 索引并打印统计")
    parser.add_argument("--render", action="store_true", help="调用占位渲染并写出 test_output/render.png")
    parser.add_argument("--render-output", type=str, default="test_output/render.png", help="渲染输出路径")
    parser.add_argument("--render-coarse-res", type=int, default=8, help="渲染使用的 coarse 网格分辨率")
    args = parser.parse_args()

    tsh = load_tetra()
    render_intrinsic = None
    render_extrinsic = None
    render_W = None
    render_H = None
    cams = []
    coarse_tuple = None
    if args.cam_list:
        from scene.dataset_readers import sceneLoadTypeCallbacks
        scene = sceneLoadTypeCallbacks["Colmap"](Path(args.cam_list), "images", False)
        print(f"[test.py] loaded scene from {args.cam_list}")
        # debug()
        sparse = pre_resterization.pre_rasterize_scene(scene, voxel_size=args.voxel_size, resize=args.resize)
        cams, _, _ = load_cameras_from_file(Path(args.cam_list))
        # if cams:
        #     render_intrinsic = cams[0]["intrinsic"]
        #     render_extrinsic = cams[0]["extrinsic"]
        #     render_W = int(cams[0]["intrinsic"][0, 2] * 2)
        #     render_H = int(cams[0]["intrinsic"][1, 2] * 2)
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
        # 记录单张相机参数用于渲染
        render_intrinsic = np.load(args.intrinsic) if args.intrinsic.endswith(".npy") else np.loadtxt(args.intrinsic)
        render_extrinsic = np.load(args.extrinsic) if args.extrinsic.endswith(".npy") else np.loadtxt(args.extrinsic)
        render_W = int(round(render_intrinsic[0, 2] * 2))
        render_H = int(round(render_intrinsic[1, 2] * 2))
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
        # 构造合成相机参数
        fx = args.res[0] / (2 * np.tan(np.deg2rad(args.fov) / 2))
        fy = args.res[1] / (2 * np.tan(np.deg2rad(args.fov) / 2))
        cx = args.res[0] * 0.5
        cy = args.res[1] * 0.5
        render_intrinsic = np.array([[fx, 0, cx],
                                     [0, fy, cy],
                                     [0,  0,  1]], dtype=np.float32)
        cam_pos = np.array([0.0, 0.0, args.cam_z], dtype=np.float32)
        cam_rot = np.eye(3, dtype=np.float32)
        render_extrinsic = np.eye(4, dtype=np.float32)
        render_extrinsic[:3, :3] = cam_rot.T
        render_extrinsic[:3, 3] = -cam_rot.T @ cam_pos
        render_W = args.res[0]
        render_H = args.res[1]

    # 初始化顶点特征 [N,4]：sigma/opacity=1，RGB=1.0
    vertex_features = tsh.initialize_vertex_features(
        sparse,
        4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fill_value=0.1,
        as_int=False,
    )
    if vertex_features.numel() > 0:
        vertex_features[:, 0] = 1.0
        vertex_features[:, 1:] = torch.randn_like(vertex_features[:, 1:]) * 0.5 + 0.5

    out_path = Path(args.output)
    sparse.write_tetra_ply(str(out_path))
    print(f"[test.py] sparse grid built: num_voxels={sparse.cell_count()}, num_vertices={sparse.vertex_count()}")
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
        print(f"[test.py] sparse grid update: num_voxels={sparse.cell_count()}, num_vertices={sparse.vertex_count()}")
        if feats.numel() > 0:
            sample = feats[0].detach().cpu().numpy()
            print(f"[test.py] sample[0]: {sample}")

    if args.dump_coarse:
        want_cuda = False
        if args.coarse_device == "cuda":
            want_cuda = torch.cuda.is_available()
            if not want_cuda:
                print("[test.py] requested cuda coarse grid but CUDA not available, falling back to CPU")
        elif args.coarse_device == "auto":
            want_cuda = torch.cuda.is_available()

        coarse = tsh.build_coarse_occupancy(sparse, res=args.coarse_res, on_cuda=want_cuda)
        nonzero = int(coarse.sum().item())
        print(f"[test.py] coarse occupancy built: shape={tuple(coarse.shape)}, device={coarse.device}, nonzero={nonzero}")

    if args.dump_coarse_index:
        want_cuda = False
        if args.coarse_device == "cuda":
            want_cuda = torch.cuda.is_available()
            if not want_cuda:
                print("[test.py] requested cuda coarse grid but CUDA not available, falling back to CPU")
        elif args.coarse_device == "auto":
            want_cuda = torch.cuda.is_available()

        occ, offsets, keys, mask = tsh.build_coarse_index(sparse, res=args.coarse_res, on_cuda=want_cuda)
        total = int(offsets[-1].item()) if offsets.numel() > 0 else 0
        nz_bricks = int(occ.sum().item())
        print(f"[test.py] coarse index built: occ_shape={tuple(occ.shape)}, offsets_shape={tuple(offsets.shape)}, keys_shape={tuple(keys.shape)}, mask_shape={tuple(mask.shape)}, device={occ.device}")
        print(f"[test.py]   bricks with voxels={nz_bricks}, total voxels listed={total}")

    if args.render:
        if not torch.cuda.is_available():
            print("[test.py] CUDA not available, skip rendering")
            return 0
        out_dir = Path(args.render_output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        if (render_intrinsic is None or render_extrinsic is None or render_W is None or render_H is None) and not cams:
            print("[test.py] render_intrinsic/extrinsic not available, skip rendering")
            return 0

        if coarse_tuple is None:
            coarse_tuple = tsh.build_coarse_index(sparse, res=args.render_coarse_res, on_cuda=False)
        vertex_dense = tsh.build_dense_vertex_grids_from_features(sparse, vertex_features)
        
        if cams:
            for cam in cams:
                render_intrinsic = cam["intrinsic"]
                render_extrinsic = cam["extrinsic"]
                render_W = int(render_intrinsic[0, 2] * 2)
                render_H = int(render_intrinsic[1, 2] * 2)
                intr_t = torch.tensor(render_intrinsic, device="cuda", dtype=torch.float32)
                extr_t = torch.tensor(render_extrinsic, device="cuda", dtype=torch.float32)
                # while (1):
                time0 = time.time()
                # debug()
                colors = tsh.rasterize_with_coarse(
                    sparse,
                    intr_t,
                    extr_t,
                    coarse_tuple,
                    int(render_H),
                    int(render_W),
                    coarse_res=args.render_coarse_res,
                    vertex_features=None,
                    vertex_dense=vertex_dense,
                )
                time1 = time.time()
                print("[test.py] rasterization timing test: {:.2f} ms".format((time1 - time0) * 1000))
                out_path = out_dir / (Path(cam["image"]).stem + "_render.png")
                img = colors.detach().cpu().clamp(0, 1).numpy()
                img = (img * 255).astype(np.uint8).reshape(render_H, render_W, 3)
                cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"[test.py] rendered image saved to {out_path}")
            return 0
        else:
            H = render_H
            W = render_W
            intr_t = torch.tensor(render_intrinsic, device="cuda", dtype=torch.float32)
            extr_t = torch.tensor(render_extrinsic, device="cuda", dtype=torch.float32)
            colors = tsh.rasterize_with_coarse(
                sparse,
                intr_t,
                extr_t,
                coarse_tuple,
                int(H),
                int(W),
                coarse_res=args.render_coarse_res,
                vertex_features=None,
                vertex_dense=vertex_dense,
            )
            sparse.write_tetra_ply(str(out_path))
            img = colors.detach().cpu().clamp(0, 1).numpy()
            img = (img * 255).astype(np.uint8).reshape(H, W, 3)
            cv2.imwrite(str(args.render_output), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"[test.py] rendered image saved to {args.render_output}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
