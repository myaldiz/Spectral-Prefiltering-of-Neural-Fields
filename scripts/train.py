import hydra
import torch
from pathlib import Path
from omegaconf import OmegaConf

from spnf.trainer import Trainer


@hydra.main(config_path="../spnf/configs", config_name="train", version_base=None)
def main(cfg):
    """
    Main training script for SPNF.
    """
    print(OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the trainer
    trainer = Trainer(cfg).to(device)

    # Run the training
    trainer.fit()

    # Create output directory for checkpoints
    checkpoint_dir = Path(cfg.trainer.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save the final model state
    state_dict = trainer.state_dict()
    checkpoint_path = checkpoint_dir / "final.pth"
    torch.save(state_dict, str(checkpoint_path))
    print(f"Saved final checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
