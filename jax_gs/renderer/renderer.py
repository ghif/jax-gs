import jax.numpy as jnp
from jax_gs.core.gaussians import Gaussians
from jax_gs.core.camera import Camera
from jax_gs.renderer.projection import project_gaussians
from jax_gs.renderer.rasterizer import get_tile_interactions, render_tiles, TILE_SIZE

def render(gaussians: Gaussians, camera: Camera, background=jnp.array([0.0, 0.0, 0.0])):
    """
    Main entry point for rendering.
    """
    # 1. Project Gaussians to 2D
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, camera)
    
    # 2. Extract colors (SH Degree 0 for now)
    # SH_DC = (R - 0.5) / 0.28209 -> R = SH_DC * 0.28209 + 0.5
    colors = gaussians.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5
    colors = jnp.clip(colors, 0.0, 1.0)
    
    # 3. Sort interactions
    sorted_tile_ids, sorted_gaussian_ids, _ = get_tile_interactions(
        means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE
    )
    
    # 4. Rasterize tiles
    image = render_tiles(
        means2D, cov2D, gaussians.opacities, colors,
        sorted_tile_ids, sorted_gaussian_ids,
        camera.H, camera.W, TILE_SIZE, background
    )
    
    return image
