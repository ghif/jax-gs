import jax
import optax
from functools import partial
from jax_gs.renderer.renderer import render
from jax_gs.training.losses import l1_loss, d_ssim_loss
from jax_gs.core.camera import Camera
import jax.numpy as jnp

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def train_step(state, target_image, w2c, camera_static, optimizer, use_pallas=False, backend="gpu"):
    """
    Standard training step for 3DGS.
    Args:
        state: (params, opt_state)
        target_image: (H, W, 3)
        w2c: (4, 4)
        camera_static: (W, H, fx, fy, cx, cy)
        optimizer: optax optimizer
        use_pallas: Use Pallas backend for rasterization
        backend: Accelerator backend for Pallas (gpu or tpu)
    Returns:
        (next_params, next_opt_state), loss, metrics
    """
    params, opt_state = state
    W, H, fx, fy, cx, cy = camera_static
    
    # Reconstruct Camera object inside JIT    
    camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
    
    lambda_ssim = 0.2

    def loss_fn(p):
        image, extras = render(p, camera, use_pallas=use_pallas, backend=backend)
        l1 = l1_loss(image, target_image)
        d_ssim = d_ssim_loss(image, target_image)

        total_loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * d_ssim

        metrics = {
            "l1": l1,
            "ssim": 1.0 - d_ssim * 2.0
        }

        return total_loss, metrics
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    
    return (next_params, next_opt_state), loss, metrics

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3, 4, 5, 6))
def train_step_parallel(state, target_image, w2c, camera_static, optimizer, use_pallas=False, backend="gpu"):
    """
    Data-parallel training step for 3DGS using pmap.
    Each device processes one image, gradients are averaged across devices.
    """
    params, opt_state = state
    W, H, fx, fy, cx, cy = camera_static
    
    # Reconstruct Camera object inside pmap
    camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
    
    lambda_ssim = 0.2

    def loss_fn(p):
        image, extras = render(p, camera, use_pallas=use_pallas, backend=backend)
        l1 = l1_loss(image, target_image)
        d_ssim = d_ssim_loss(image, target_image)

        total_loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * d_ssim

        metrics = {
            "l1": l1,
            "ssim": 1.0 - d_ssim * 2.0
        }
            
        return total_loss, metrics
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Average gradients across all devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    # Average loss and metrics for logging consistency
    loss = jax.lax.pmean(loss, axis_name='batch')
    metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), metrics)
    
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    
    return (next_params, next_opt_state), loss, metrics
