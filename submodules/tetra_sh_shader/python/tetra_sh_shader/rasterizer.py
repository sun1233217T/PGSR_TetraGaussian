from __future__ import annotations

import torch

from tetra_sh_shader_cpp import (
    rasterize_image,
    rasterize_image_with_index,
    rasterize_image_with_index_dense,
    rasterize_image_with_index_dense_backward,
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
    def forward(
        ctx,
        grid,
        intrinsic,
        extrinsic,
        coarse_tuple,
        height: int,
        width: int,
        coarse_res: int = 8,
        vertex_features=None,
        sigma=None,
        color=None,
        valid=None,
        xyz=None,
    ):
        _, offsets, keys, mask = coarse_tuple
        ctx.grid = grid
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.coarse_res = int(coarse_res)
        use_dense = sigma is not None and color is not None and sigma.numel() > 0 and color.numel() > 0
        ctx.use_dense = use_dense
        if use_dense:
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
        # 保存张量供反向使用（确保都是 Tensor）
        saved = [
            intrinsic,
            extrinsic,
            offsets,
            keys,
            mask,
            sigma if sigma is not None else torch.Tensor(),
            color if color is not None else torch.Tensor(),
            valid if valid is not None else torch.Tensor(),
        ]
        ctx.save_for_backward(*saved)
        return colors

    @staticmethod
    def backward(ctx, grad_output):
        intrinsic, extrinsic, offsets, keys, mask, sigma, color, valid = ctx.saved_tensors
        grid = ctx.grid
        height = ctx.height
        width = ctx.width
        coarse_res = ctx.coarse_res

        grad_grid = None
        grad_intrinsic = None
        grad_extrinsic = None
        grad_coarse_tuple = None
        grad_height = None
        grad_width = None
        grad_coarse_res = None
        grad_vertex_features = None
        grad_sigma = None
        grad_color = None
        grad_valid = None
        grad_xyz = None

        if ctx.use_dense:
            grads = rasterize_image_with_index_dense_backward(
                grid,
                intrinsic,
                extrinsic,
                offsets,
                keys,
                mask,
                sigma,
                color,
                valid,
                grad_output,
                int(height),
                int(width),
                int(coarse_res),
            )
            grad_sigma, grad_color = grads

        return (
            grad_grid,
            grad_intrinsic,
            grad_extrinsic,
            grad_coarse_tuple,
            grad_height,
            grad_width,
            grad_coarse_res,
            grad_vertex_features,
            grad_sigma,
            grad_color,
            grad_valid,
            grad_xyz,
        )


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
    if vertex_dense is not None:
        sigma, color, valid, xyz = vertex_dense
        return VoxelRasterizeWithCoarseFunction.apply(
            grid, intrinsic, extrinsic, coarse_tuple, height, width, coarse_res, vertex_features, sigma, color, valid, xyz
        )
    return VoxelRasterizeWithCoarseFunction.apply(
        grid, intrinsic, extrinsic, coarse_tuple, height, width, coarse_res, vertex_features, None, None, None, None
    )
