from __future__ import annotations

from typing import Union, Sequence, Any, Callable, Sequence

import math
import random
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_to_tensors(data: Any, func: Callable[[torch.Tensor], Any]) -> Any:
    """
    Recursively traverses lists, tuples, and dicts. 
    Applies `func` to any torch.Tensor found.
    """
    if isinstance(data, dict):
        return {k: apply_to_tensors(v, func) for k, v in data.items()}
    elif isinstance(data, list):
        return [apply_to_tensors(v, func) for v in data]
    elif isinstance(data, tuple):
        return tuple(apply_to_tensors(v, func) for v in data)
    elif isinstance(data, torch.Tensor):
        return func(data)
    else:
        # Return other types (str, int, float, None) as-is
        return data


def to_py(t):
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu()
                return t.item() if t.numel() == 1 else t.tolist()
            return t


def make_coord_grid(ndim: int, 
                    resolution: Union[int, Sequence[int]], 
                    bounds: Union[float, Sequence[float]]=1.0, 
                    align_corners: bool = False, 
                    device: torch.device = torch.device("cpu"),
                    reversed_order: bool = True,
                    ) -> torch.Tensor:
    """
    Creates an n-dimensional coordinate grid with explicit dimensionality.
    
    Args:
        ndim: The number of dimensions (e.g., 2, 3).
        resolution: int (same for all dims) or Sequence (res_0, res_1, ...).
        bounds: float (implies [-b, b] for all dims) or Sequence (min_0...min_n, max_0...max_n).
    """
    # 1. Validate and Normalize Resolution
    if isinstance(resolution, int):
        resolution = [resolution] * ndim
    elif len(resolution) != ndim:
        raise ValueError(f"Resolution length ({len(resolution)}) must match ndim ({ndim}).")
        
    # 2. Validate and Normalize Bounds
    # Expected pattern for sequence: [min_0, ..., min_n, max_0, ..., max_n]
    if isinstance(bounds, (float, int)):
        mins = [-float(bounds)] * ndim
        maxs = [float(bounds)] * ndim
    else:
        if len(bounds) != 2 * ndim:
            raise ValueError(f"Bounds length ({len(bounds)}) must match 2 * ndim ({2 * ndim}).")
        mins = bounds[:ndim]
        maxs = bounds[ndim:]
        
    if reversed_order and ndim >= 2:
        mins = mins[::-1]
        maxs = maxs[::-1]

    # 3. Generate coordinates
    coords = []
    for i in range(ndim):
        res = resolution[i]
        mn, mx = mins[i], maxs[i]
        
        if not align_corners:
            step = (mx - mn) / res
            # Pixel centers
            coord = torch.linspace(mn + step/2, mx - step/2, res, device=device)
        else:
            # Corners
            coord = torch.linspace(mn, mx, res, device=device)
        coords.append(coord)
        
    # 4. Meshgrid and Stack
    grids = torch.meshgrid(*coords, indexing='ij')
    if reversed_order:
        grids = grids[::-1]
    return torch.stack(grids, dim=-1)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save an image tensor in [-1, 1] range to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = tensor.detach().cpu().clamp(-1, 1)
    if data.ndim == 3 and data.shape[0] in (1, 3):
        data = data.permute(1, 2, 0)
    elif data.ndim == 2:
        data = data[..., None]
    data = ((data + 1.0) * 0.5).clip(0, 1)
    data = (data * 255).byte().numpy()
    iio.imwrite(path, data)


def WarmUpScheduler(end_warm=500, end_iter=300000, curve="cosine", warmup_curve="linear", decay_start=None, decay_approach_curve="constant", min_val=0.0, max_val=1.0):
    if decay_start is None:
        decay_start = end_warm
    def scheduler(step):
        multiplier = 1.0
        if step < end_warm:
            if warmup_curve == "zero":
                multiplier = 0.0
            elif warmup_curve == "one":
                multiplier = 1.0
            elif warmup_curve == "linear":
                multiplier = step / end_warm
            else:
                raise ValueError("Unknown warmup curve type")
        elif step < decay_start:
            if decay_approach_curve == "constant":
                multiplier = 1.0
            elif decay_approach_curve == "linear":
                multiplier = 1.0 - (decay_start - step) / (decay_start - end_warm)
        else:
            if curve == "cosine":
                multiplier = 0.5 * (1.0 + np.cos(np.pi * (step - decay_start) / (end_iter - decay_start)))
            elif curve == "linear":
                multiplier = 1.0 - (step - decay_start) / (end_iter - decay_start)
            elif curve == "constant":
                multiplier = 1.0
            elif curve == "exponential":
                lr = max_val * (min_val / max_val) ** ((step - decay_start) / (end_iter - decay_start))
                multiplier = (lr - min_val) / (max_val - min_val)
            else:
                raise ValueError("Unknown curve type")
        multiplier = np.clip(multiplier, 0.0, 1.0)
        return min_val + multiplier * (max_val - min_val)
        
    return scheduler