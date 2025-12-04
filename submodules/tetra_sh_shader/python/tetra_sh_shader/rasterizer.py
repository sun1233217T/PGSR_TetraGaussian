from __future__ import annotations

import torch

from tetra_sh_shader_cpp import (
    rasterize_image,
    rasterize_image_with_index,
    rasterize_image_with_index_dense,
    rasterize_image_with_index_dense_backward,
    rasterize_image_with_index_packed,
    rasterize_image_with_index_packed_backward,
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
        packed_meta=None,
    ):
        _, offsets, keys, mask = coarse_tuple
        ctx.grid = grid
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.coarse_res = int(coarse_res)
        use_packed = packed_meta is not None
        use_dense = not use_packed and sigma is not None and color is not None and sigma.numel() > 0 and color.numel() > 0
        ctx.use_dense = use_dense
        ctx.use_packed = use_packed
        if use_packed:
            brick_starts, brick_shapes, brick_offsets = packed_meta
            colors = rasterize_image_with_index_packed(
                grid,
                intrinsic,
                extrinsic,
                offsets,
                keys,
                mask,
                sigma if sigma is not None else torch.Tensor(),
                color if color is not None else torch.Tensor(),
                valid if valid is not None else torch.Tensor(),
                brick_starts,
                brick_shapes,
                brick_offsets,
                int(height),
                int(width),
                int(coarse_res),
            )
        elif use_dense:
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
        saved = [
            intrinsic,
            extrinsic,
            offsets,
            keys,
            mask,
            sigma if sigma is not None else torch.Tensor(),
            color if color is not None else torch.Tensor(),
            valid if valid is not None else torch.Tensor(),
            torch.Tensor() if not use_packed else packed_meta[0],
            torch.Tensor() if not use_packed else packed_meta[1],
            torch.Tensor() if not use_packed else packed_meta[2],
        ]
        ctx.save_for_backward(*saved)
        return colors

    @staticmethod
    def backward(ctx, grad_output):
        intrinsic, extrinsic, offsets, keys, mask, sigma, color, valid, b_starts, b_shapes, b_offsets = ctx.saved_tensors
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
        grad_packed = None

        if ctx.use_packed:
            grads = rasterize_image_with_index_packed_backward(
                grid,
                intrinsic,
                extrinsic,
                offsets,
                keys,
                mask,
                sigma,
                color,
                valid,
                b_starts,
                b_shapes,
                b_offsets,
                grad_output,
                int(height),
                int(width),
                int(coarse_res),
            )
            grad_sigma, grad_color = grads
        elif ctx.use_dense:
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
            grad_packed,
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
    if vertex_dense is not None and len(vertex_dense) == 4:
        sigma, color, valid, xyz = vertex_dense
        return VoxelRasterizeWithCoarseFunction.apply(
            grid, intrinsic, extrinsic, coarse_tuple, height, width, coarse_res, vertex_features, sigma, color, valid, xyz, None
        )
    if vertex_dense is not None and len(vertex_dense) == 6:
        sigma, color, valid, brick_starts, brick_shapes, brick_offsets = vertex_dense
        packed_meta = (brick_starts, brick_shapes, brick_offsets)
        return VoxelRasterizeWithCoarseFunction.apply(
            grid, intrinsic, extrinsic, coarse_tuple, height, width, coarse_res, vertex_features, sigma, color, valid, None, packed_meta
        )
    return VoxelRasterizeWithCoarseFunction.apply(
        grid, intrinsic, extrinsic, coarse_tuple, height, width, coarse_res, vertex_features, None, None, None, None, None
    )
