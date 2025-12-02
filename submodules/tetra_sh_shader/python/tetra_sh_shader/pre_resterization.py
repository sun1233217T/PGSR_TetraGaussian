from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from tetra_sh_shader_cpp import (
    Vec3,
    VoxelGrid,
    grid_init,
    image_pre_resterization,
    grad_sparsilization,
)
from utils.graphics_utils import getWorld2View2


def rays_from_image_mask(
    image_path: str,
    mask_path: Optional[str],
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    resize: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成相机射线，并返回 mask>0 (正) 与 mask==0 (负) 的两组射线。
    Returns:
        origins_pos, dirs_world_pos, origins_neg, dirs_world_neg
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


def cameras_from_scene(scene) -> Tuple[list, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build camera list and optional bbox (origin, size) from a Colmap scene object.
    """
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


def pre_rasterize_scene(
    scene,
    voxel_size: float,
    resize: int = 50,
) -> VoxelGrid:
    """
    Args:
        scene: sceneLoadTypeCallbacks["Colmap"](...) 返回的 scene 对象
        voxel_size: 体素边长
        resize: rays_from_image_mask 的下采样因子
    Returns:
        稀疏 VoxelGrid
    """
    cams, bbox_origin, bbox_size = cameras_from_scene(scene)
    all_o_pos, all_d_pos, all_o_neg, all_d_neg = [], [], [], []
    for cam in cams:
        o_pos, d_pos, o_neg, d_neg = rays_from_image_mask(
            cam["image"], cam["mask"], cam["intrinsic"], cam["extrinsic"],
            resize=resize,
        )
        all_o_pos.append(o_pos)
        all_d_pos.append(d_pos.reshape(-1, 1, 3))
        all_o_neg.append(o_neg)
        all_d_neg.append(d_neg.reshape(-1, 1, 3))

    o_pos = np.concatenate(all_o_pos, axis=0) if all_o_pos else np.empty((0, 3), dtype=np.float64)
    d_pos = np.concatenate(all_d_pos, axis=0) if all_d_pos else np.empty((0, 3), dtype=np.float64)
    o_neg = np.concatenate(all_o_neg, axis=0) if all_o_neg else np.empty((0, 3), dtype=np.float64)
    d_neg = np.concatenate(all_d_neg, axis=0) if all_d_neg else np.empty((0, 3), dtype=np.float64)

    grid = VoxelGrid()
    if bbox_origin is not None and bbox_size is not None:
        origin_np = bbox_origin
        size_np = bbox_size
    else:
        origin_np = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        size_np = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    origin_vec = Vec3(*origin_np.tolist())
    size_vec = Vec3(*size_np.tolist())
    grid.build_dense(origin_vec, size_vec, voxel_size)

    # 反削除：mask==0，tolerant=0，invert=True
    grid_init(grid)
    if o_neg.size > 0:
        image_pre_resterization(grid, o_neg, d_neg)
    grid = grad_sparsilization(grid, 0, True)

    # 正削除：mask>0，tolerant=2，invert=False
    grid_init(grid)
    if o_pos.size > 0:
        image_pre_resterization(grid, o_pos, d_pos)
    sparse = grad_sparsilization(grid, 2, False)
    return sparse
