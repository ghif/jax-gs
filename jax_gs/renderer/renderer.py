import jax.numpy as jnp
from jax_gs.core.gaussians import Gaussians
from jax_gs.core.camera import Camera
from jax_gs.renderer.projection import project_gaussians
from jax_gs.renderer.rasterizer import get_tile_interactions, render_tiles, TILE_SIZE

try:
    from jax_gs.renderer.rasterizer_pallas import render_tiles_pallas
    HAS_PALLAS = True
except ImportError:
    HAS_PALLAS = False

def render(gaussians: Gaussians, camera: Camera, background=jnp.array([0.0, 0.0, 0.0]), 
           use_pallas: bool = False, backend: str = "gpu"):
    """
    Main entry point for rendering.

    Args:
        gaussians: Gaussians dataclass
        camera: Camera dataclass
        background: Background color
        use_pallas: Use Pallas backend
        backend: Accelerator backend for Pallas (gpu or tpu)
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
    if use_pallas and HAS_PALLAS:
        # Rasterize tiles using Pallas
        image = render_tiles_pallas(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background,
            backend=backend
        )
    else:
        # Rasterize tiles using pure JAX
        image = render_tiles(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background
        )
    return image, {}
