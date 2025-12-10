from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Simple feed-forward network with ReLU activations."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, num_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        # Initialize the final layer bias to zero
        nn.init.zeros_(self.net[-1].bias)
        
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SPNFModel(nn.Module):
    """Minimal SPNF model: Fourier encoder followed by an MLP regressor."""
    def __init__(
        self,
        input_dim: int=2,
        output_dim: int=3,
        encoder: torch.nn.Module=None,
        mlp: torch.nn.Module=None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = encoder
        self.mlp = mlp
        
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
        
    def forward(self, batch=None, X=None, covariances=None, filter_type="gaussian"):
        if batch is not None:
            X = batch["coords"]
            covariances = batch.get("covariances", None)
            filter_type = batch.get("filter_type", filter_type)

        encoded = self.encoder(X, covariances, filter_type=filter_type)
        filtered_signal = self.mlp(encoded)
        return {"filtered_signal": filtered_signal}

    @staticmethod
    def loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return dict(mse_loss=F.mse_loss(pred["filtered_signal"], batch["gt_signal"]))