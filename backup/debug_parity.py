import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.core.camera import Camera
from jax_gs.renderer.renderer import render
from jax_gs.renderer.projection import project_gaussians

def test_full_render_parity():
    p = jnp.array([[0.0, 0.0, 5.0]]) 
    c = jnp.array([[1.0, 0.0, 0.0]])
    g_single = init_gaussians_from_pcd(p, c).replace(
        scales=jnp.array([[-1.0, -1.0, -1.0]]),
        opacities=jnp.array([[10.0]])
    )
    cam_s = Camera(W=128, H=128, fx=100.0, fy=100.0, cx=64.0, cy=64.0, W2C=jnp.eye(4), full_proj=jnp.eye(4))
    
    # 1. Project
    means2D, cov2D, radii, valid_mask, depths = project_gaussians(g_single, cam_s)
    
    # 2. Extract MLX inv_cov
    import mlx.core as mx
    m_cov2D = mx.array(np.array(cov2D))
    det = m_cov2D[:, 0, 0] * m_cov2D[:, 1, 1] - m_cov2D[:, 0, 1]**2
    det = mx.maximum(det, 1e-6)
    m_inv_cov2D = mx.stack([
        mx.stack([m_cov2D[:, 1, 1] / det, -m_cov2D[:, 0, 1] / det], axis=-1),
        mx.stack([-m_cov2D[:, 1, 0] / det, m_cov2D[:, 0, 0] / det], axis=-1)
    ], axis=-2)
    
    # 3. Extract JAX inv_cov (manual for check)
    det_j = cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1]**2
    det_j = jnp.maximum(det_j, 1e-6)
    j_inv_cov2D = jnp.stack([
        jnp.stack([cov2D[:, 1, 1] / det_j, -cov2D[:, 0, 1] / det_j], axis=-1),
        jnp.stack([-cov2D[:, 1, 0] / det_j, cov2D[:, 0, 0] / det_j], axis=-1)
    ], axis=-2)
    
    print(f"JAX inv_cov[0]:\n{j_inv_cov2D[0]}")
    print(f"MLX inv_cov[0]:\n{np.array(m_inv_cov2D[0])}")
    
    # Render
    img_jax = np.array(render(g_single, cam_s, use_mlx=False))
    img_mlx = np.array(render(g_single, cam_s, use_mlx=True))
    
    print(f"JAX [64,64]: {img_jax[64,64]}")
    print(f"MLX [64,64]: {img_mlx[64,64]}")

if __name__ == "__main__":
    test_full_render_parity()
