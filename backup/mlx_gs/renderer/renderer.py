import mlx.core as mx
from mlx_gs.renderer.projection import project_gaussians
from mlx_gs.renderer.rasterizer import get_tile_interactions, render_tiles

TILE_SIZE = 16

def render_mlx(params: dict, camera_dict: dict, background=mx.array([0.0, 0.0, 0.0])) -> mx.array:
    """
    Standard MLX rendering pipeline.
    """
    # 1. Project
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(params, camera_dict)
    
    # 2. Extract colors (SH-to-RGB)
    colors = params["sh_coeffs"][:, 0, :] * 0.28209479177387814 + 0.5
    colors = mx.clip(colors, 0.0, 1.0)
    
    # 3. Rasterize
    sorted_tile_ids, sorted_gaussian_ids, _ = get_tile_interactions(
        means2D, radii, valid_mask, depths, camera_dict["H"], camera_dict["W"], TILE_SIZE
    )
    
    image = render_tiles(
        means2D, cov2D, params["opacities"], colors,
        sorted_tile_ids, sorted_gaussian_ids,
        camera_dict["H"], camera_dict["W"], TILE_SIZE, background
    )
    
    return image
