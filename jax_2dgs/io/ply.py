import jax.numpy as jnp
from jax_gs.core.gaussians import Gaussians
from jax_2dgs.core.gaussians_2d import Gaussians2D
from jax_gs.io.ply import save_ply, load_ply

def save_ply_2d(path, gaussians: Gaussians2D):
    """
    Save 2D Gaussians to a PLY file compatible with 3DGS viewers 
    by appending a small epsilon scale for the third dimension.

    Args:
        path: Path to save the PLY file
        gaussians: Gaussians2D dataclass
    """
    print(f"Saving 2D PLY to {path}...")
    
    # Convert 2D scales to 3D for compatibility with existing viewers
    num_points = gaussians.means.shape[0]
    scales_3d = jnp.concatenate([gaussians.scales, jnp.full((num_points, 1), -10.0)], axis=-1)
    
    # Use standard 3D Gaussians container for saving
    g3d = Gaussians(
        means=gaussians.means,
        scales=scales_3d,
        quaternions=gaussians.quaternions,
        opacities=gaussians.opacities,
        sh_coeffs=gaussians.sh_coeffs
    )
    save_ply(path, g3d)

def load_ply_2d(path):
    """
    Load 2D Gaussians from a PLY file.
    
    Args:
        path: Path to the PLY file
    Returns:
        gaussians: Gaussians2D dataclass
    """
    g3d = load_ply(path)
    # Drop the third scale dimension
    return Gaussians2D(
        means=g3d.means,
        scales=g3d.scales[:, :2],
        quaternions=g3d.quaternions,
        opacities=g3d.opacities,
        sh_coeffs=g3d.sh_coeffs
    )
