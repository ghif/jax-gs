import jax
import optax
from functools import partial
from jax_gs.renderer.renderer import render
from jax_gs.training.losses import l1_loss
from jax_gs.core.camera import Camera
import jax.numpy as jnp

@partial(jax.jit, static_argnums=(3, 4))
def train_step(state, target_image, w2c, camera_static, optimizer):
    """
    Standard training step.
    Args:
        state: (params, opt_state)
        target_image: (H, W, 3)
        w2c: (4, 4)
        camera_static: (W, H, fx, fy, cx, cy)
        optimizer: optax optimizer
    Returns:
        (next_params, next_opt_state), loss 
    """
    params, opt_state = state
    W, H, fx, fy, cx, cy = camera_static
    
    # Reconstruct Camera object inside JIT    
    camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
    
    def loss_fn(p):
        image = render(p, camera)
        return l1_loss(image, target_image)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    
    return (next_params, next_opt_state), loss
