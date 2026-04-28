import jax
import optax
from functools import partial
from jax_gs.renderer.renderer import render
from jax_gs.training.losses import l1_loss, d_ssim_loss, depth_distortion_loss, normal_consistency_loss
from jax_gs.core.camera import Camera
import jax.numpy as jnp

@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def train_step(state, target_image, w2c, camera_static, optimizer, use_pallas=False, mode="3dgs", backend="gpu"):
    """
    Standard training step supporting both 3DGS and 2DGS.
    Args:
        state: (params, opt_state)
        target_image: (H, W, 3)
        w2c: (4, 4)
        camera_static: (W, H, fx, fy, cx, cy)
        optimizer: optax optimizer
        use_pallas: Use Pallas backend for rasterization
        mode: Rendering mode ('3dgs' or '2dgs')
    Returns:
        (next_params, next_opt_state), loss, metrics
    """
    params, opt_state = state
    W, H, fx, fy, cx, cy = camera_static
    
    # Reconstruct Camera object inside JIT    
    camera = Camera(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy, W2C=w2c, full_proj=jnp.eye(4))
    
    lambda_ssim = 0.2
    lambda_distortion = 0.001
    lambda_normal = 0.01 # Adjusted for stability
    
    def loss_fn(p):
        image, extras = render(p, camera, use_pallas=use_pallas, mode=mode, backend=backend)
        l1 = l1_loss(image, target_image)
        d_ssim = d_ssim_loss(image, target_image)
        
        total_loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * d_ssim
        
        metrics = {
            "l1": l1,
            "ssim": 1.0 - d_ssim * 2.0
        }
        
        if mode == "2dgs":
            # Re-enable all regularization losses with stable implementations
            l_dist = depth_distortion_loss(extras["depth"], extras["depth_sq"])
            l_normal = normal_consistency_loss(extras["normals"], extras["depth"], camera)

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
