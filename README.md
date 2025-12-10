# Spectral Prefiltering of Neural Fields (SPNF)

This repository hosts a minimal, Hydra-driven implementation of the Spectral Prefiltering of Neural Fields project. The focus is on a clean training pipeline, lightweight dependencies, and a compact `spnf` package that can be installed directly.

## Quickstart

```bash
pip install -e .
python scripts/train.py
```

The default config trains on a small synthetic gradient image so you can verify the pipeline without extra assets. To train on your own image, set `data.image_path` in `configs/train.yaml` or override it from the CLI:

```bash
python scripts/train.py data.image_path=/path/to/image.png
```

## Configuration

Training is fully driven by Hydra via `configs/train.yaml`. Key options:
- `data`: image path, coordinate bounds, Monte Carlo samples for smoothing, and log-scale sampling range.
- `model`: Fourier encoder width, frequency spread, and MLP size.
- `optim`: learning rate and weight decay.
- `train`: total steps, batch size, eval cadence, checkpointing, and output locations.

Outputs (checkpoints and visualizations) are written under `outputs/` by default.

## Package Layout

- `spnf/data.py`: image loading, grid-based field, optional Monte Carlo smoothing, and a batch generator.
- `spnf/encoding.py`: spectral Fourier encoder with scale-aware attenuation.
- `spnf/mlp.py`: small feed-forward network.
- `spnf/model.py`: model definition and loss.
- `spnf/trainer.py`: lightweight training loop with evaluation and PNG dumps.
- `scripts/train.py`: Hydra entrypoint.

## Notes

- All tensors are normalized to `[-1, 1]`.
- If no image path is provided, a procedural gradient is used for sanity checks.
- Dependencies are minimal; no Docker or multi-GPU plumbing is included to keep the code base lean.
