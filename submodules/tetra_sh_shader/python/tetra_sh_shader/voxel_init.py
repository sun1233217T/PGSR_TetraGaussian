from __future__ import annotations

from typing import Union

import torch

from tetra_sh_shader_cpp import (
    init_vertex_tensor,
    ensure_active_cells_from_vertex_attr,
)


def initialize_vertex_features(
    grid,
    feature_dim: int,
    device: Union[str, torch.device] = "cuda",
    fill_value: float = 0.0,
    as_int: bool = False,
) -> torch.Tensor:
    """
    Two-pass initialization:
      1) create a (N,1) int tensor filled with ones on CPU and attach to vertices
      2) promote all cells that have at least one attributed corner
      3) create the real (N,C) tensor on the requested device and attach to vertices

    Returns:
        torch.Tensor: the (N, C) feature tensor backing all vertex attrs.
    """
    # Pass 1: tag active vertices with int ones on CPU
    init_vertex_tensor(grid, 1, on_cuda=False, fill_ones=True, as_int=True)
    ensure_active_cells_from_vertex_attr(grid)

    # Pass 2: allocate real features on target device
    target = torch.device(device)
    on_cuda = target.type == "cuda"
    feats = init_vertex_tensor(
        grid,
        feature_dim,
        on_cuda=on_cuda,
        fill_ones=False,
        as_int=as_int,
    )
    if fill_value != 0 and not as_int:
        feats.fill_(fill_value)
    return feats
