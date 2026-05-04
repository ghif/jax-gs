import jax.numpy as jnp
from jax_gs.core.gaussians import Gaussians
from jax_gs.core.camera import Camera
from jax_gs.renderer.projection import project_gaussians
from jax_gs.renderer.rasterizer import get_tile_interactions, render_tiles, TILE_SIZE

def render(gaussians: Gaussians, camera: Camera, background=jnp.array([0.0, 0.0, 0.0]), 
           fast_tpu_rasterizer: bool = False):
    """
    Main entry point for rendering.

    Args:
        gaussians: Gaussians dataclass
        camera: Camera dataclass
        background: Background color
        fast_tpu_rasterizer: Use optimized JAX scan rasterizer for TPU
    Returns:
        image: Rendered image
        extras: Optional dictionary with auxiliary maps (depth, normals, etc.)
    """
    # --- 3DGS Pipeline ---
    # 1. Project Gaussians to 2D
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, camera)
    
    colors = gaussians.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5
    colors = jnp.clip(colors, 0.0, 1.0)
    
    # 2. Sort interactions
    sorted_tile_ids, sorted_gaussian_ids, n_interactions = get_tile_interactions(
        means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE
    )
    
    # 3. Rasterize tiles
    if fast_tpu_rasterizer:
        from jax_gs.renderer.rasterizer_tpu import render_tiles_tpu
        image = render_tiles_tpu(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, background
        )
        extras = {}
    else:
        # Rasterize tiles using pure JAX standard implementation
        image = render_tiles(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background
        )
        extras = {}
    return image, extras
