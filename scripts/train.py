#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from spnf.data import FieldBatchGenerator, GridField, MonteCarloField, load_image
from spnf.model import ModelConfig
from spnf.trainer import OptimConfig, TrainConfig, Trainer


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.train.device)

    image = load_image(cfg.data.image_path, device=device)
    base_field = GridField(image, bounds=cfg.data.bounds, padding_mode=cfg.data.padding_mode)
    field = MonteCarloField(base_field, samples=cfg.data.mc_samples) if cfg.data.mc_samples > 1 else base_field

    data = FieldBatchGenerator(
        field=field,
        bounds=cfg.data.bounds,
        log_scale_range=tuple(cfg.data.log_scale_range),
        device=device,
    )

    model_cfg = ModelConfig(
        coords=cfg.model.coords,
        fourier_features=cfg.model.fourier_features,
        freq_std=cfg.model.freq_std,
        hidden_dim=cfg.model.hidden_dim,
        hidden_layers=cfg.model.hidden_layers,
        output_dim=cfg.model.output_dim,
    )
    optim_cfg = OptimConfig(lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    train_cfg = TrainConfig(
        steps=cfg.train.steps,
        batch_size=cfg.train.batch_size,
        eval_every=cfg.train.eval_every,
        checkpoint_every=cfg.train.checkpoint_every,
        eval_scale=cfg.train.eval_scale,
        eval_resolution=cfg.train.eval_resolution,
        checkpoint_dir=cfg.train.checkpoint_dir,
        log_dir=cfg.train.log_dir,
        seed=cfg.train.seed,
        device=cfg.train.device,
    )

    trainer = Trainer(model_cfg, optim_cfg, train_cfg, data)
    trainer.fit()


if __name__ == "__main__":
    main()
