from typing import Union, Sequence
from pathlib import Path
from tqdm.notebook import tqdm
import hydra
import torch

import spnf
from spnf.utils import set_seed, make_coord_grid, apply_to_tensors, to_py
from spnf.sample import (
    rand_ortho, logrand, construct_covariance, get_deltas
)


class Trainer(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        set_seed(cfg.trainer.seed)
        self.cfg = cfg
        self.model = hydra.utils.instantiate(cfg.model)
        self.data = hydra.utils.instantiate(cfg.data)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=self.optimizer)
        if cfg.get("tensorboard", None) is not None:
            self.writer = hydra.utils.instantiate(cfg.tensorboard)
        self.global_step = 0
        
    @property
    def device(self):
        return self.model.device
    
    def state_dict(self):
        state_dict = dict(
            model=self.model.state_dict(),
            global_step=self.global_step,
            optimizer=self.optimizer.state_dict(),
        )
        if self.scheduler:
            state_dict["scheduler"] = self.scheduler.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.global_step = state_dict["global_step"]
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
    
    @torch.no_grad()
    def generate_data_train(self, coords=None, eigenvalues=None, eigenvectors=None, perturb_kernel=None) -> dict[str, torch.Tensor]:
        batch = {}
        if coords is None:
            coords_orig = None
            coords = self.data.generate_coords(self.cfg.trainer.batch_size)
        else:
            coords_orig = coords
            coords = coords.view(-1, self.model.input_dim)
        
        batch_size = coords.shape[0]
        is_multiscale = isinstance(self.cfg.trainer.mc_samples_train, str) and self.cfg.trainer.mc_samples_train == "perturb"
        is_multiscale = is_multiscale or (isinstance(self.cfg.trainer.mc_samples_train, int) and self.cfg.trainer.mc_samples_train > 0)
        
        if is_multiscale:
            if eigenvalues is None and eigenvectors is None:
                eigenvalues = logrand(
                    self.cfg.trainer.covariance_eigenvalue_logrange[0],
                    self.cfg.trainer.covariance_eigenvalue_logrange[1],
                    (batch_size, self.model.input_dim),
                    device=coords.device
                )
                eigenvectors = rand_ortho(
                    self.model.input_dim,
                    batch_size,
                    device=coords.device,
                )
            else:
                eigenvalues = eigenvalues.view(-1, self.model.input_dim)
                eigenvectors = eigenvectors.view(-1, self.model.input_dim, self.model.input_dim)
                
            batch["covariances"] = construct_covariance(eigenvectors, eigenvalues)
            perturb_kernel = self.cfg.trainer.perturb_kernel if perturb_kernel is None else perturb_kernel
        
        
        if is_multiscale:
            if isinstance(self.cfg.trainer.mc_samples_train, int):
                num_mc_samples = self.cfg.trainer.mc_samples_train
                signal_average = torch.zeros(batch_size, self.model.output_dim, device=coords.device, dtype=coords.dtype)
                for _ in range(num_mc_samples):
                    deltas = get_deltas(eigenvectors, eigenvalues, perturb_kernel)
                    signal = self.data(coords + deltas)
                    signal_average += signal
                signal_average /= num_mc_samples
                batch["gt_signal"] = signal_average
            elif self.cfg.trainer.mc_samples_train == "perturb":
                batch["gt_signal"] = self.data(coords)
                deltas = get_deltas(eigenvectors, eigenvalues, perturb_kernel)
                coords = coords + deltas
                batch["deltas"] = deltas
            else:
                raise ValueError(f"Unknown mc_samples_train method: {self.cfg.trainer.mc_samples_train}")
        else:
            batch["gt_signal"] = self.data(coords)
                
        if coords_orig is not None:
            batch["coords"] = coords_orig
            batch["gt_signal"] = batch["gt_signal"].view(*coords_orig.shape[:-1], -1)
        else:
            batch["coords"] = coords
        
        return batch
    
    @torch.no_grad()
    def generate_data_eval(self, coords=None, resolution=None, eigenvalues=None, eigenvectors=None, perturb_kernel=None, bounds=1.0) -> dict[str, torch.Tensor]:
        batch = {}
        
        # Create the coords
        if coords is None:
            resolution = self.cfg.trainer.eval_resolution if resolution is None else resolution
            coords = make_coord_grid(
                self.model.input_dim,
                resolution=resolution,
                bounds=bounds,
                device=self.device,
            )

        # Coords flatten
        original_shape = coords.shape[:-1]
        coords = coords.view(-1, self.model.input_dim)
        batch_size = coords.shape[0]
        batch["coords"] = coords
        batch["original_shape"] = original_shape
        
        # Eigenvalues and eigenvectors flatten
        if eigenvalues is not None and eigenvectors is not None:
            eigenvalues = eigenvalues.view(-1, self.model.input_dim)
            eigenvectors = eigenvectors.view(-1, self.model.input_dim, self.model.input_dim)
        
        # Calculate the GT signal
        is_multiscale = eigenvalues is not None and eigenvectors is not None
        if is_multiscale:
            # Calculate number of MC samples
            mc_samples_eval = self.cfg.trainer.mc_samples_eval
            if mc_samples_eval == "dynamic":
                q = (1e5 * eigenvalues).clamp(10, 1000).prod(dim=-1).max()
                n_samples = round(20 * q.sqrt().item())
                mc_samples_eval = max(min(n_samples, self.cfg.trainer.eval_max_mc_samples), self.cfg.trainer.eval_min_mc_samples)
            
            # Check eigenvalues and eigenvectors shape
            if eigenvalues.shape[0] != batch_size:
                eigenvalues = eigenvalues.expand(batch_size, -1)
                eigenvectors = eigenvectors.expand(batch_size, -1, -1)
                batch["eigenvalues"] = eigenvalues
                batch["eigenvectors"] = eigenvectors
            
            perturb_kernel = self.cfg.trainer.perturb_kernel if perturb_kernel is None else perturb_kernel
            batch["covariances"] = construct_covariance(eigenvectors, eigenvalues)
            gt_signal = torch.zeros(batch_size, self.model.output_dim, device=coords.device, dtype=coords.dtype)
            for _ in range(mc_samples_eval):
                deltas = get_deltas(eigenvectors, eigenvalues, perturb_kernel)
                signal = self.data(coords + deltas)
                gt_signal += signal
            gt_signal /= mc_samples_eval
        else:
            gt_signal = self.data(coords)
        
        batch["gt_signal"] = gt_signal.view(*original_shape, -1)
            
        return batch

    def train_step(self) -> dict[str, torch.Tensor]:
        batch = self.generate_data_train()
        
        pred = self.model(batch)
        loss = self.model.loss(pred, batch)

        loss_sum = sum(loss.values())
        self.optimizer.zero_grad()
        loss_sum.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        return loss
    
    def generate_grid_data(self, resolution: Union[int, Sequence[int]]=512, eigenvalues=None, eigenvectors=None, covariances=None, bounds=1.0, **kwargs) -> dict[str, torch.Tensor]:
        # Setup grid resolution
        if isinstance(resolution, int):
            grid_res = (resolution,) * self.model.input_dim
        else:
            grid_res = tuple(resolution)
        
        batch = {}
        
        # Create coordinate grid
        coord_grid = make_coord_grid(
            self.model.input_dim,
            resolution=grid_res,
            bounds=bounds,
            device=self.device,
        )
        
        # Flatten coordinate grid
        total_pixels = coord_grid.shape[:-1].numel() 
        flattened_coords = coord_grid.view(total_pixels, self.model.input_dim)
        batch["coords"] = flattened_coords
        
        # Eigenvalues and eigenvectors flatten
        if eigenvalues is not None and eigenvectors is not None:
            eigenvalues = eigenvalues.view(-1, self.model.input_dim)
            eigenvectors = eigenvectors.view(-1, self.model.input_dim, self.model.input_dim)
            batch["covariances"] = construct_covariance(eigenvectors, eigenvalues)
            
        if covariances is not None:
            covariances = covariances.view(-1, self.model.input_dim, self.model.input_dim)
            batch["covariances"] = covariances
        
        # Inference
        with torch.no_grad():
            output = self.model.forward(batch, **kwargs)
            
        output["coords"] = coord_grid
            
        def reshape_if_matching(t: torch.Tensor):
            if t.shape[0] == total_pixels:
                new_shape = grid_res + t.shape[1:]
                return t.view(new_shape)
            return t
            
        return apply_to_tensors(output, reshape_if_matching)

    def fit(self, num_steps=None, no_tqdm=False) -> dict:
        # 1. Compile train step
        if self.cfg.trainer.compile_train_step:
            train_step = torch.compile(self.train_step)
        else:
            train_step = self.train_step
        
        if num_steps is None:
            num_steps = self.cfg.trainer.steps

        pbar = tqdm(total=num_steps, desc="Training", dynamic_ncols=True, disable=no_tqdm)
        
        stats_history = []
        for _ in range(num_steps):
            stats = train_step()
            self.global_step += 1
            
            # Convert tensors to python scalars/lists
            step_stats = {k: to_py(v) for k, v in stats.items()}
            stats_history.append(step_stats)

            # Update progress bar (using the raw tensor items for display is fine/fast)
            pbar.update(1)
            pbar.set_postfix({k: f"{v:.4f}"  for k, v in step_stats.items() if isinstance(v, float)})

        pbar.close()
        
        # Concatenate (Transpose)
        if stats_history:
            final_stats = {k: [step[k] for step in stats_history] for k in stats_history[0]}
        else:
            final_stats = {}
            
        return final_stats
    
