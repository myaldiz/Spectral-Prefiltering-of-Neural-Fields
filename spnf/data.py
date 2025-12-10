from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import imageio.v3 as iio
import torch
import torch.nn.functional as F
from torch import nn

from .utils import make_coord_grid


def load_image(
    path: Optional[Path | str],
    device: Optional[torch.device] = None,
    resize: Optional[int | Tuple[int, int]] = None,
    resize_mode: Optional[str] = "area",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    """
    Load an image as a float tensor in [-1, 1] range.
    If the path is None or does not exist, a simple synthetic gradient is returned.
    Optionally resize to a target resolution (H, W) with a chosen interpolation mode.
    """
    if path is None or not Path(path).exists():
        axis = torch.linspace(0, 1, 256)
        gx, gy = torch.meshgrid(axis, axis, indexing="xy")
        img = torch.stack([gx, gy, torch.ones_like(gx) * 0.5], dim=0)
    else:
        raw = torch.as_tensor(iio.imread(path)).float()
        if raw.ndim == 2:
            raw = raw[..., None]
        if raw.shape[-1] > 4:
            raw = raw[..., :3]
        if raw.max() > 1.0:
            max_val = 255.0 if raw.max() <= 255 else 65535.0
            raw = raw / max_val
        img = raw.permute(2, 0, 1)

    if resize is not None:
        size = (resize, resize) if isinstance(resize, int) else tuple(resize)
        size = tuple(int(s) for s in size)
        if resize_mode == "area":
            img = F.interpolate(img[None], size=size, mode=resize_mode)[0]
        else:
            img = F.interpolate(img[None], size=size, mode=resize_mode, align_corners=align_corners)[0]

    img = img * 2.0 - 1.0
    if device:
        img = img.to(device)
    return img


class GridField(nn.Module):
    """Continuous 2D field backed by an image tensor."""

    def __init__(self, grid: torch.Tensor, bounds: float = 1.0, padding_mode: str = "reflection", align_corners: bool = False) -> None:
        super().__init__()
        if grid.ndim == 3:
            grid = grid[None]
        self.register_buffer("grid", grid)
        self.padding_mode = padding_mode
        self.coords = 2
        self.channels = grid.shape[1]
        self.align_corners = align_corners
        self.bounds = bounds

    @property
    def device(self) -> torch.device:
        return self.grid.device

    def generate_coords(self, n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        bounds = self.bounds if self.bounds is not None else 1.0
        device = device or self.device
        return (torch.rand(n, self.coords, device=device) * 2.0 - 1.0) * bounds

    def forward(self, coords: torch.Tensor, covariances: Optional[torch.Tensor] = None, mode="bilinear") -> torch.Tensor:
        coords = coords.view(1, -1, 1, 2)
        samples = F.grid_sample(
            self.grid,
            coords,
            mode=mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return samples[0, :, :, 0].transpose(0, 1)
