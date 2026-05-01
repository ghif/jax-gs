import mlx.core as mx
import mlx.optimizers as opt
import numpy as np
import os
import random
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from PIL import Image

from mlx_gs.renderer.renderer import render_mlx
from mlx_gs.training.losses import l1_loss, d_ssim_loss
from mlx_gs.io.ply import save_ply

def default_loss_fn(params: dict, camera_dict: dict, target: mx.array) -> mx.array:
    """
    Default loss function: 0.8 * L1 + 0.2 * SSIM.
    """
    image = render_mlx(params, camera_dict)
    l1 = l1_loss(image, target)
    d_ssim = d_ssim_loss(image, target)
    return 0.8 * l1 + 0.2 * d_ssim

class Trainer:
    def __init__(
        self,
        params: dict,
        lr: float = 1e-3,
        loss_fn = default_loss_fn
    ):
        self.params = params
        self.optimizer = opt.Adam(learning_rate=lr)
        self.loss_fn = loss_fn
        self.grad_fn = mx.value_and_grad(self.loss_fn)

    def train_step(self, camera_dict: dict, target: mx.array) -> float:
        """
        Perform a single training step.
        """
        loss, grads = self.grad_fn(self.params, camera_dict, target)
        
        if mx.isnan(loss):
            return float('nan')
            
        self.optimizer.update(self.params, grads)
        mx.eval(self.params, self.optimizer.state)
        return loss.item()

    def train(
        self,
        cameras: List[dict],
        targets: List[mx.array],
        num_iterations: int,
        output_dir: str,
        save_interval: int = 100,
        log_interval: int = 10
    ):
        """
        Full training loop.
        """
        os.makedirs(output_dir, exist_ok=True)
        progress_dir = os.path.join(output_dir, "progress")
        os.makedirs(progress_dir, exist_ok=True)

        pbar = tqdm(range(num_iterations))
        for i in pbar:
            idx = random.randint(0, len(cameras) - 1)
            cam = cameras[idx]
            target = targets[idx]
            
            loss = self.train_step(cam, target)
            
            if np.isnan(loss):
                print(f"NaN loss detected at iteration {i}. Aborting.")
                break

            if i % log_interval == 0:
                pbar.set_description(f"Loss: {loss:.4f}")

            if i % save_interval == 0:
                # Save progress image
                img = render_mlx(self.params, cameras[0])
                img_np = np.array(img)
                Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8)).save(
                    os.path.join(progress_dir, f"progress_{i:04d}.png")
                )
                # Save PLY
                save_ply(os.path.join(output_dir, f"splats_{i:04d}.ply"), self.params)

        # Final save
        save_ply(os.path.join(output_dir, "splats_final.ply"), self.params)
