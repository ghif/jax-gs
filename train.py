import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
from renderer import Camera, render
from gaussians import Gaussians, init_gaussians_from_pcd

def l1_loss(pred, target):
    return jnp.mean(jnp.abs(pred - target))

def train_step(state, target_image, camera, optimizer):
    params, opt_state = state
    
    def loss_fn(p):
        image = render(p, camera)
        return l1_loss(image, target_image)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    
    return (next_params, next_opt_state), loss

def run_training():
    # 1. Setup Data
    W, H = 128, 128
    num_points = 500
    
    # Target: random Gaussians
    true_points = np.random.uniform(-0.5, 0.5, (num_points, 3))
    true_colors = np.random.uniform(0, 1, (num_points, 3))
    true_gaussians = init_gaussians_from_pcd(jnp.array(true_points), jnp.array(true_colors))
    
    cam = Camera(
        W=W, H=H,
        fx=100.0, fy=100.0,
        cx=W/2, cy=H/2,
        W2C=jnp.eye(4),
        full_proj=jnp.eye(4)
    )
    
    target_image = render(true_gaussians, cam)
    
    # 2. Initialize Model
    init_points = np.random.uniform(-1, 1, (num_points, 3))
    init_colors = np.random.uniform(0, 1, (num_points, 3))
    gaussians = init_gaussians_from_pcd(jnp.array(init_points), jnp.array(init_colors))
    
    # 3. Optimizer
    # Separate learning rates for different parameters
    # For now, keep it simple
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(gaussians)
    
    state = (gaussians, opt_state)
    
    # 4. Training Loop
    pbar = tqdm(range(1000))
    for i in pbar:
        state, loss = train_step(state, target_image, cam, optimizer)
        if i % 10 == 0:
            pbar.set_description(f"Loss: {loss:.4f}")
            
    print("Training complete!")

if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'cpu')
    run_training()
