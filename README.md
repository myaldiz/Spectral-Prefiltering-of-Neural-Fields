# Spectral Prefiltering of Neural Fields (SPNF)

This repository hosts a minimal, Hydra-driven implementation of the Spectral Prefiltering of Neural Fields project. The focus is on a clean training pipeline, lightweight dependencies, and a compact `spnf` package that can be installed directly.

## Quickstart

1.  **Install dependencies and the package in editable mode:**
    ```bash
    pip install -e .
    ```

2.  **Run training with default settings:**
    ```bash
    python scripts/train.py
    ```
    The default config trains on a small synthetic gradient image. Checkpoints and logs will be saved to a directory specified in `spnf/configs/train.yaml` under `hydra.run.dir`.

## Command-line Training

You can override any configuration from `spnf/configs/train.yaml` directly from the command line.

**Example 1: Train on your own image**
```bash
python scripts/train.py data.grid.path=/path/to/your/image.png
```

**Example 2: Train a larger model for more steps**
```bash
python scripts/train.py trainer.steps=20000 model.mlp.num_layers=4 model.mlp.hidden_dim=512
```

**Example 3: Change learning rate and batch size**
```bash
python scripts/train.py optimizer.lr=1e-4 trainer.batch_size=65536
```

## Configuration

Training is fully driven by Hydra via `configs/train.yaml`. All parameters can be overriden from the command line. Key options:
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


## Useful commands 
Command to convert videos to keynote friendly format:
```bash
ffmpeg -i ours_vis.mp4 \
  -c:v libx265 \
  -pix_fmt yuv420p \
  -crf 20 \
  -preset slow \
  -tune grain \
  -tag:v hvc1 \
  output_vis.mov
```