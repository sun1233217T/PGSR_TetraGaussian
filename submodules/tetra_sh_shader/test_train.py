"""
Minimal training smoke test using the differentiable rasterizer.

Usage (example):
CUDA_VISIBLE_DEVICES=1 python submodules/tetra_sh_shader/test_train.py \
    --cam-list ~/data/DTU/scan24/ --voxel-size 0.02 --render --dump-coarse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from tetra_sh_shader import pre_resterization
from tetra_sh_shader import (
    build_coarse_index,
    initialize_vertex_features,
    rasterize_with_coarse,
    build_dense_vertex_grids_from_features,
)
from tetra_sh_shader.pre_resterization import cameras_from_scene
from scene.dataset_readers import sceneLoadTypeCallbacks


def load_tetra():
    import importlib
    import sys

    try:
        import tetra_sh_shader as tsh  # type: ignore
        return tsh
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        pkg_root = repo_root / "submodules" / "tetra_sh_shader" / "python"
        sys.path.insert(0, str(pkg_root))
        import tetra_sh_shader as tsh  # type: ignore
        return importlib.reload(tsh)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-list", type=str, required=True, help="Colmap 数据集路径")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="体素边长")
    parser.add_argument("--resize", type=int, default=50, help="预栅格化下采样")
    parser.add_argument("--render-coarse-res", type=int, default=8, help="coarse 网格分辨率")
    parser.add_argument("--iters", type=int, default=100, help="训练步数")
    parser.add_argument("--lr", type=float, default=1e-2, help="学习率")
    parser.add_argument("--render", action="store_true", help="训练结束后写出渲染结果")
    parser.add_argument("--dump-coarse", action="store_true", help="打印 coarse 统计")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this training test.")
    device = torch.device("cuda")

    # 加载 Colmap 场景与相机
    scene = sceneLoadTypeCallbacks["Colmap"](Path(args.cam_list), "images", False)
    cams, _, _ = cameras_from_scene(scene)
    if not cams:
        raise RuntimeError("No cameras found in the dataset.")
    cam_data = []
    for cam in cams:
        intr_np = cam["intrinsic"].astype(np.float32)
        extr_np = cam["extrinsic"].astype(np.float32)
        target_img = cv2.imread(cam["image"], cv2.IMREAD_COLOR)
        if target_img is None:
            raise FileNotFoundError(f"无法读取图像 {cam['image']}")
        H, W = target_img.shape[:2]
        target = torch.from_numpy(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)).float() / 255.0
        cam_data.append({
            "intr": intr_np,
            "ext": extr_np,
            "target": target,  # CPU 张量，按需搬到 GPU
            "H": int(H),
            "W": int(W),
        })
    H = cam_data[0]["H"]
    W = cam_data[0]["W"]

    # 预栅格化得到稀疏网格
    sparse = pre_resterization.pre_rasterize_scene(scene, voxel_size=args.voxel_size, resize=args.resize)

    # coarse 结构
    coarse_tuple = build_coarse_index(sparse, res=args.render_coarse_res, on_cuda=False)
    if args.dump_coarse:
        occ, offsets, keys, mask = coarse_tuple
        total = int(offsets[-1].item()) if offsets.numel() > 0 else 0
        print(f"[train] coarse: bricks={occ.numel()}, voxels listed={total}, nz bricks={int(occ.sum().item())}")

    # 初始化顶点特征并转稠密
    vertex_features = initialize_vertex_features(
        sparse,
        4,
        device="cpu",
        fill_value=0.1,
        as_int=False,
    )
    if vertex_features.numel() > 0:
        vertex_features[:, 0] = 1.0
    vertex_dense = build_dense_vertex_grids_from_features(sparse, vertex_features)
    sigma, color, valid, xyz = vertex_dense
    sigma = sigma.to(device).requires_grad_()
    color = color.to(device).requires_grad_()
    valid = valid.to(device)
    xyz = xyz.to(device)
    vertex_dense = (sigma, color, valid, xyz)

    optimizer = torch.optim.Adam([sigma, color], lr=args.lr)

    print(f"[train] start, iters={args.iters}, cams={len(cam_data)}, HxW~={H}x{W}")
    for it in range(args.iters):
        optimizer.zero_grad()
        loss_acc = 0.0
        for cam in cam_data:
            intr_t = torch.tensor(cam["intr"], device=device)
            extr_t = torch.tensor(cam["ext"], device=device)
            target = cam["target"].to(device)
            colors = rasterize_with_coarse(
                sparse,
                intr_t,
                extr_t,
                coarse_tuple,
                cam["H"],
                cam["W"],
                coarse_res=args.render_coarse_res,
                vertex_features=None,
                vertex_dense=vertex_dense,
            )
            loss = (colors - target).pow(2).mean()
            loss.backward()
            loss_acc += loss.item()
        loss_acc /= float(len(cam_data))
        optimizer.step()
        print(f"[train] iter {it+1}/{args.iters} mean_loss={loss_acc:.6f}")

    if args.render:
        out_dir = Path("test_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(cam_data)):
            with torch.no_grad():
                cam0 = cam_data[i]
                intr_t = torch.tensor(cam0["intr"], device=device)
                extr_t = torch.tensor(cam0["ext"], device=device)
                colors = rasterize_with_coarse(
                    sparse,
                    intr_t,
                    extr_t,
                    coarse_tuple,
                    cam0["H"],
                    cam0["W"],
                    coarse_res=args.render_coarse_res,
                    vertex_features=None,
                    vertex_dense=vertex_dense,
                )
            img = colors.detach().cpu().clamp(0, 1).numpy()
            img = (img * 255).astype(np.uint8)
            out_path = out_dir / "train_render_{:02d}.png".format(i)
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[train] rendered image saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
