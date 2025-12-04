"""
Lightweight helpers for the tetra_sh_shader C++ sample.

Installing this package does not build the C++ target; it only provides
paths you can use from Python after configuring CMake yourself.
"""

from __future__ import annotations

import pathlib

import torch  # ensure torch shared libs are loaded before importing the extension

from tetra_sh_shader_cpp import (
    Vec3,
    Vec3i,
    VoxelKey,
    Attribute,
    IntAttribute,
    Vertex,
    TetraMesh,
    VoxelCell,
    VoxelGrid,
    grid_init,
    image_pre_resterization,
    grad_sparsilization,
    init_vertex_tensor,
    attach_vertex_tensor,
    ensure_active_cells_from_vertex_attr,
    build_coarse_occupancy,
    build_coarse_index,
    rasterize_image,
    rasterize_image_with_index,
    rasterize_image_with_index_dense,
    rasterize_image_with_index_dense_backward,
    rasterize_image_with_index_packed,
    rasterize_image_with_index_packed_backward,
    build_dense_vertex_grids_from_features,
    build_packed_dense_vertex_grids_from_features,
)
from .pre_resterization import (
    rays_from_image_mask,
    cameras_from_scene,
    pre_rasterize_scene,
)
from .rasterizer import (
    rasterize,
    rasterize_with_coarse,
    VoxelRasterizeFunction,
    VoxelRasterizeWithCoarseFunction,
)
from .voxel_init import initialize_vertex_features


def default_build_dir() -> pathlib.Path:
    """
    Return the default CMake build directory inside the submodule.
    Note: __file__ is inside python/tetra_sh_shader/, so we need to go up
    two levels to the submodule root.
    """
    return pathlib.Path(__file__).resolve().parents[2] / "build"


def sample_binary_path(name: str = "tetra_sh_shader") -> pathlib.Path:
    """
    Point to the expected sample binary location.
    Useful for packaging or invoking the executable from Python.
    """
    return default_build_dir() / name


__all__ = [
    "default_build_dir",
    "sample_binary_path",
    "Vec3",
    "Vec3i",
    "VoxelKey",
    "Attribute",
    "IntAttribute",
    "Vertex",
    "TetraMesh",
    "VoxelCell",
    "VoxelGrid",
    "grid_init",
    "image_pre_resterization",
    "grad_sparsilization",
    "init_vertex_tensor",
    "attach_vertex_tensor",
    "ensure_active_cells_from_vertex_attr",
    "build_coarse_occupancy",
    "build_coarse_index",
    "rasterize_image",
    "rasterize_image_with_index",
    "rasterize_image_with_index_dense",
    "rasterize_image_with_index_dense_backward",
    "rasterize_image_with_index_packed",
    "rasterize_image_with_index_packed_backward",
    "build_dense_vertex_grids_from_features",
    "build_packed_dense_vertex_grids_from_features",
    "rasterize",
    "rasterize_with_coarse",
    "VoxelRasterizeFunction",
    "VoxelRasterizeWithCoarseFunction",
    "initialize_vertex_features",
    "rays_from_image_mask",
    "cameras_from_scene",
    "pre_rasterize_scene",
]
