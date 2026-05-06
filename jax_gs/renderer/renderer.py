import jax.numpy as jnp
from jax_gs.core.gaussians import Gaussians
from jax_gs.core.camera import Camera
from jax_gs.renderer.projection import project_gaussians
from jax_gs.renderer.rasterizer import (
    BLOCK_SIZE,
    OFFSET_SIZE,
    TILE_SIZE,
    get_tile_interactions,
    render_tiles,
)
from jax_gs.renderer.sh import eval_sh

def render(gaussians: Gaussians, camera: Camera, background=None,
           fast_tpu_rasterizer: bool = False, mask=None, sh_degree: int = 0):
    """
    Main entry point for rendering.

    Args:
        gaussians: Gaussians dataclass
        camera: Camera dataclass
        background: Background color
        fast_tpu_rasterizer: Use optimized JAX scan rasterizer for TPU
        mask: Optional mask for active Gaussians
        sh_degree: Current SH degree to use for color computation (0-3)
    Returns:
        image: Rendered image
        extras: Optional dictionary with auxiliary maps (depth, normals, etc.)
    """
    if background is None:
        background = jnp.zeros((3,), dtype=jnp.float32)

    # --- 3DGS Pipeline ---
    # 1. Project Gaussians to 2D
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, camera, mask=mask)

    # 2. View-dependent color computation
    # Calculate viewing directions (normalized vector from Gaussian to camera center)
    view_dirs = gaussians.means - camera.center
    view_dirs = view_dirs / jnp.linalg.norm(view_dirs, axis=-1, keepdims=True)

    # Evaluate SH (adds view-dependent specularities)
    colors = eval_sh(sh_degree, gaussians.sh_coeffs, view_dirs)

    # Sigmoid / Offset to ensure [0, 1]
    # Standard 3DGS adds 0.5 to the SH evaluation result (which covers the DC component offset)
    colors = colors + 0.5
    colors = jnp.clip(colors, 0.0, 1.0)

    # 3. Sort interactions

    sorted_tile_ids, sorted_gaussian_ids, n_interactions = get_tile_interactions(
        means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE
    )

    num_tiles_x = (camera.W + TILE_SIZE - 1) // TILE_SIZE
    num_tiles_y = (camera.H + TILE_SIZE - 1) // TILE_SIZE
    num_tiles = num_tiles_x * num_tiles_y
    tile_boundaries = jnp.searchsorted(sorted_tile_ids, jnp.arange(num_tiles + 1))
    tile_counts = tile_boundaries[1:] - tile_boundaries[:-1]

    min_x = jnp.clip((means2D[:, 0] - radii), 0, camera.W - 1)
    max_x = jnp.clip((means2D[:, 0] + radii), 0, camera.W - 1)
    min_y = jnp.clip((means2D[:, 1] - radii), 0, camera.H - 1)
    max_y = jnp.clip((means2D[:, 1] + radii), 0, camera.H - 1)
    tile_min_x = (min_x // TILE_SIZE).astype(jnp.int32)
    tile_max_x = (max_x // TILE_SIZE).astype(jnp.int32)
    tile_min_y = (min_y // TILE_SIZE).astype(jnp.int32)
    tile_max_y = (max_y // TILE_SIZE).astype(jnp.int32)
    on_screen = (means2D[:, 0] + radii > 0) & (means2D[:, 0] - radii < camera.W) & \
                (means2D[:, 1] + radii > 0) & (means2D[:, 1] - radii < camera.H)
    capped_span = ((tile_max_x - tile_min_x + 1) > OFFSET_SIZE) | \
                  ((tile_max_y - tile_min_y + 1) > OFFSET_SIZE)
    radius_cap_violations = jnp.sum(capped_span & valid_mask & on_screen)
    overflow = jnp.maximum(tile_counts - BLOCK_SIZE, 0)
    
    extras = {
        "radii": radii,
        "valid_mask": valid_mask,
        "n_interactions": n_interactions,
        "mean_interactions_per_tile": n_interactions / jnp.maximum(num_tiles, 1),
        "max_interactions_per_tile": jnp.max(tile_counts),
        "overflow_tiles": jnp.sum(tile_counts > BLOCK_SIZE),
        "overflow_interactions": jnp.sum(overflow),
        "radius_cap_violations": radius_cap_violations,
    }
    
    # 3. Rasterize tiles
    if fast_tpu_rasterizer:
        from jax_gs.renderer.rasterizer_tpu import render_tiles_tpu
        image = render_tiles_tpu(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, background
        )
    else:
        # Rasterize tiles using pure JAX standard implementation
        image = render_tiles(
            means2D, cov2D, gaussians.opacities, colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, background
        )
    return image, extras
