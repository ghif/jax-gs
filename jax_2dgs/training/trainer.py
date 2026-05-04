import jax
import optax
from functools import partial
from jax_2dgs.renderer.renderer import render
from jax_gs.training.losses import l1_loss, d_ssim_loss
from jax_2dgs.training.losses import depth_distortion_loss, normal_consistency_loss
from jax_gs.core.camera import Camera
import jax.numpy as jnp

def train_step_internal(state, target_image, w2c, camera_static, optimizer):
    """
    Internal training step for 2DGS, suitable for use inside scan/pmap.
    """
    params, opt_state = state
    W, H, fx, fy, cx, cy = camera_static
    
    # Reconstruct Camera object inside pmap
    camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
    
    lambda_ssim = 0.2
    lambda_distortion = 0.0001
    lambda_normal = 0.0001

    def loss_fn(p):
        image, extras = render(p, camera)
        l1 = l1_loss(image, target_image)
        d_ssim = d_ssim_loss(image, target_image)

        total_loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * d_ssim

        metrics = {
            "l1": l1,
            "ssim": 1.0 - d_ssim * 2.0
        }

        # Unpack to allow XLA to optimize memory lifetimes more aggressively
        depth = extras.get("depth")
        depth_sq = extras.get("depth_sq")
        accum_weight = extras.get("accum_weight")
        normals = extras.get("normals")
        
        l_dist = depth_distortion_loss(depth, depth_sq, accum_weight)
        l_normal = normal_consistency_loss(normals, depth, camera)

        total_loss = total_loss + lambda_distortion * l_dist + lambda_normal * l_normal
        metrics.update({
            "dist_loss": l_dist,
            "normal_loss": l_normal
        })
            
        return total_loss, metrics
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Average gradients across all devices (SPMD)
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    # Average loss and metrics for logging consistency
    loss = jax.lax.pmean(loss, axis_name='batch')
    metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), metrics)
    
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    
    return (next_params, next_opt_state), loss, metrics

@partial(jax.jit, static_argnums=(3, 4))
def train_step(state, target_image, w2c, camera_static, optimizer):
    """
    Standard training step for 2DGS.
    Args:
        state: (params, opt_state)
        target_image: (H, W, 3)
        w2c: (4, 4)
        camera_static: (W, H, fx, fy, cx, cy)
        optimizer: optax optimizer
    Returns:
        (next_params, next_opt_state), loss, metrics
    """
    params, opt_state = state
    W, H, fx, fy, cx, cy = camera_static
    
    # Reconstruct Camera object inside JIT    
    camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
    
    lambda_ssim = 0.2
    lambda_distortion = 0.0001
    lambda_normal = 0.0001

    def loss_fn(p):
        image, extras = render(p, camera)
        l1 = l1_loss(image, target_image)
        d_ssim = d_ssim_loss(image, target_image)

        total_loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * d_ssim

        metrics = {
            "l1": l1,
            "ssim": 1.0 - d_ssim * 2.0
        }

        # Unpack to allow XLA to optimize memory lifetimes more aggressively
        depth = extras.get("depth")
        depth_sq = extras.get("depth_sq")
        accum_weight = extras.get("accum_weight")
        normals = extras.get("normals")
        
        # Pass accum_weight to the stabilized loss functions
        l_dist = depth_distortion_loss(depth, depth_sq, accum_weight)
        l_normal = normal_consistency_loss(normals, depth, camera)

        total_loss = total_loss + lambda_distortion * l_dist + lambda_normal * l_normal
        metrics.update({
            "dist_loss": l_dist,
            "normal_loss": l_normal
        })

        return total_loss, metrics
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    
    return (next_params, next_opt_state), loss, metrics
