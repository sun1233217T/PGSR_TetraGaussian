from __future__ import annotations

import torch

from tetra_sh_shader_cpp import (
    rasterize_image,
    rasterize_image_with_index,
    rasterize_image_with_index_dense,
)


class VoxelRasterizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, intrinsic, extrinsic, height: int, width: int, coarse_res: int = 8, vertex_features=None):
        colors = rasterize_image(
            grid, intrinsic, extrinsic,
            int(height), int(width), int(coarse_res),
            vertex_features if vertex_features is not None else torch.Tensor(),
        )
        return colors

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: 实现反向传播（目前占位返回 None）
        return None, None, None, None, None, None, None


def rasterize(grid, intrinsic, extrinsic, height: int, width: int, coarse_res: int = 8, vertex_features=None):
    """
    Autograd-friendly wrapper around the CUDA rasterizer.
    """
    return VoxelRasterizeFunction.apply(grid, intrinsic, extrinsic, height, width, coarse_res, vertex_features)


class VoxelRasterizeWithCoarseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, intrinsic, extrinsic, coarse_tuple, height: int, width: int, coarse_res: int = 8, vertex_features=None, vertex_dense=None):
        _, offsets, keys, mask = coarse_tuple
        if vertex_dense is not None:
            sigma, color, valid, xyz = vertex_dense  # xyz currently未使用
            colors = rasterize_image_with_index_dense(
                grid,
                intrinsic,
                extrinsic,
                offsets,
                keys,
                mask,
                sigma,
                color,
                valid if valid is not None else torch.Tensor(),
                int(height),
                int(width),
                int(coarse_res),
            )
        else:
            colors = rasterize_image_with_index(
                grid,
                intrinsic,
                extrinsic,
                offsets,
                keys,
                mask,
                int(height),
                int(width),
                int(coarse_res),
                vertex_features if vertex_features is not None else torch.Tensor(),
            )
        return colors

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: 实现反向传播（目前占位返回 None）
        return None, None, None, None, None, None, None, None, None


def rasterize_with_coarse(grid, intrinsic, extrinsic, coarse_tuple, height: int, width: int, coarse_res: int = 8, vertex_features=None, vertex_dense=None):
    """
    使用预计算的 coarse 索引进行渲染，避免每帧重建 coarse 结构。

    Args:
        grid: VoxelGrid
        intrinsic: (3,3) torch/numpy
        extrinsic: (4,4) torch/numpy
        coarse_tuple: (occ, offsets, keys, mask)，通常来自 build_coarse_index
        height, width: 输出分辨率
    coarse_res: coarse 网格分辨率（需与 coarse_tuple 对应）
    """
    return VoxelRasterizeWithCoarseFunction.apply(
        grid, intrinsic, extrinsic, coarse_tuple, height, width, coarse_res, vertex_features, vertex_dense
    )
