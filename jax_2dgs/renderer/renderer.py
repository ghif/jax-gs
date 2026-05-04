import jax.numpy as jnp
from jax_2dgs.core.gaussians_2d import Gaussians2D
from jax_gs.core.camera import Camera
from jax_2dgs.renderer.projection_2d import project_gaussians_2d
from jax_gs.renderer.rasterizer import get_tile_interactions, TILE_SIZE
from jax_2dgs.renderer.rasterizer_2d import render_tiles_2d

def render(gaussians: Gaussians2D, camera: Camera, background=jnp.array([0.0, 0.0, 0.0])):
    """
    Main entry point for 2DGS rendering.

    Args:
        gaussians: Gaussians2D dataclass
        camera: Camera dataclass
        background: Background color
    Returns:
        image: Rendered image
        extras: Optional dictionary with auxiliary maps (depth, normals, etc.)
    """
    # 1. Project 2D Gaussians
    means2D, cov2D, radii, valid_mask, depths, normals = project_gaussians_2d(gaussians, camera)
    
    colors = gaussians.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5
    colors = jnp.clip(colors, 0.0, 1.0)
    
    # 2. Sort interactions
    sorted_tile_ids, sorted_gaussian_ids, n_interactions = get_tile_interactions(
        means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE
    )
    
    # 3. Rasterize tiles
    image, depth, depth_sq, normal_map, accum_weight = render_tiles_2d(
        means2D, cov2D, gaussians.opacities, colors, depths, normals,
        sorted_tile_ids, sorted_gaussian_ids,
        camera.H, camera.W, TILE_SIZE, background
    )
    
    extras = {
        "depth": depth,
        "depth_sq": depth_sq,
        "normals": normal_map,
        "accum_weight": accum_weight
    }
    return image, extras
