import jax.numpy as jnp
from jax_gs.core.gaussians import Gaussians
from jax_gs.core.gaussians_2d import Gaussians2D
from jax_gs.core.camera import Camera
from jax_gs.renderer.projection import project_gaussians
from jax_gs.renderer.projection_2d import project_gaussians_2d
from jax_gs.renderer.rasterizer import get_tile_interactions, render_tiles, TILE_SIZE
from jax_gs.renderer.rasterizer_2d import render_tiles_2d

try:
    from jax_gs.renderer.rasterizer_pallas import render_tiles_pallas
    HAS_PALLAS = True
except ImportError:
    HAS_PALLAS = False

try:
    import mlx.core as mx
    from mlx_gs.renderer.renderer import render_mlx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    from jax_gs.renderer import rasterizer_torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def render(gaussians, camera: Camera, background=jnp.array([0.0, 0.0, 0.0]), 
           use_mlx: bool = False, use_torch: bool = False, use_pallas: bool = False,
           backend: str = "gpu", mode: str = "3dgs"):
    """
    Main entry point for rendering.

    Args:
        gaussians: Gaussians or Gaussians2D dataclass
        camera: Camera dataclass
        background: Background color
        use_mlx: Use MLX backend
        use_torch: Use Torch backend
        use_pallas: Use Pallas backend
        backend: Accelerator backend for Pallas (gpu or tpu)
        mode: Rendering mode ('3dgs' or '2dgs')
    Returns:
        image: Rendered image
        extras: Optional dictionary with auxiliary maps (depth, normals, etc.)
    """
    if mode == "2dgs":
        # 1. Project 2D Gaussians
        means2D, cov2D, radii, valid_mask, depths, normals = project_gaussians_2d(gaussians, camera)
        
        colors = gaussians.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5
        colors = jnp.clip(colors, 0.0, 1.0)
        
        # 2. Sort interactions
        sorted_tile_ids, sorted_gaussian_ids, n_interactions = get_tile_interactions(
            means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE
        )
        
        # 3. Rasterize tiles (JAX only for now for 2DGS)
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

    # --- 3DGS Pipeline ---
    # 1. Project Gaussians to 2D
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(gaussians, camera)
    
    colors = gaussians.sh_coeffs[:, 0, :] * 0.28209479177387814 + 0.5
    colors = jnp.clip(colors, 0.0, 1.0)
    
    # 3. Sort and Rasterize
    if use_mlx and HAS_MLX:
        # Use centralized MLX-native rendering pipeline
        import numpy as np
        
        # Camera dict for MLX
        mlx_camera = {
            "W": int(camera.W),
            "H": int(camera.H),
            "fx": float(camera.fx),
            "fy": float(camera.fy),
            "cx": float(camera.cx),
            "cy": float(camera.cy),
            "W2C": mx.array(np.array(camera.W2C))
        }
        
        # Params dict for MLX
        params = {
            "means": mx.array(np.array(gaussians.means)),
            "scales": mx.array(np.array(gaussians.scales)),
            "quaternions": mx.array(np.array(gaussians.quaternions)),
            "opacities": mx.array(np.array(gaussians.opacities)),
            "sh_coeffs": mx.array(np.array(gaussians.sh_coeffs))
        }
        
        m_background = mx.array(np.array(background))
        
        # Render using the decoupled rendering entry point
        image_mlx = render_mlx(params, mlx_camera, m_background)
        
        # Convert back to JAX
        image = jnp.array(np.array(image_mlx))
        return image, {}
    elif use_torch and HAS_TORCH:
        # Convert JAX to Torch
        import numpy as np
        t_means2D = torch.from_numpy(np.array(means2D)).to("mps")
        t_cov2D = torch.from_numpy(np.array(cov2D)).to("mps")
        t_radii = torch.from_numpy(np.array(radii)).to("mps")
        t_valid_mask = torch.from_numpy(np.array(valid_mask)).to("mps")
        t_depths = torch.from_numpy(np.array(depths)).to("mps")
        t_opacities = torch.from_numpy(np.array(gaussians.opacities)).to("mps")
        t_colors = torch.from_numpy(np.array(colors)).to("mps")
        t_background = torch.from_numpy(np.array(background)).to("mps")
        
        sorted_tile_ids, sorted_gaussian_ids, _ = rasterizer_torch.get_tile_interactions(
            t_means2D, t_radii, t_valid_mask, t_depths, camera.H, camera.W, TILE_SIZE, device="mps"
        )
        
        image_torch = rasterizer_torch.render_tiles(
            t_means2D, t_cov2D, t_opacities, t_colors,
            sorted_tile_ids, sorted_gaussian_ids,
            camera.H, camera.W, TILE_SIZE, t_background, device="mps"
        )
        
        # Convert back to JAX
        image = jnp.array(image_torch.cpu().numpy())
        return image, {}
    else:
        # 3. Sort interactions
        sorted_tile_ids, sorted_gaussian_ids, n_interactions = get_tile_interactions(
            means2D, radii, valid_mask, depths, camera.H, camera.W, TILE_SIZE
        )
        
        if use_pallas and HAS_PALLAS:
            # 4. Rasterize tiles using Pallas
            image = render_tiles_pallas(
                means2D, cov2D, gaussians.opacities, colors,
                sorted_tile_ids, sorted_gaussian_ids,
                camera.H, camera.W, TILE_SIZE, background,
                backend=backend
            )
        else:
            # 4. Rasterize tiles using pure JAX
            image = render_tiles(
                means2D, cov2D, gaussians.opacities, colors,
                sorted_tile_ids, sorted_gaussian_ids,
                camera.H, camera.W, TILE_SIZE, background
            )
        return image, {}
