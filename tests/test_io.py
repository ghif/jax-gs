import jax
import jax.numpy as jnp
import numpy as np
import os
import pytest
from jax_gs.core.gaussians import init_gaussians_from_pcd
from jax_gs.io.ply import save_ply, load_ply

def test_ply_roundtrip(tmp_path):
    # Setup test Gaussians
    num_points = 50
    means = jnp.array(np.random.uniform(-1, 1, (num_points, 3)))
    colors = jnp.array(np.random.uniform(0, 1, (num_points, 3)))
    gaussians = init_gaussians_from_pcd(means, colors)
    
    # Randomize other params to ensure they save/load correctly
    gaussians = gaussians.replace(
        scales=jnp.array(np.random.uniform(-2, 0, (num_points, 3))),
        quaternions=jnp.array(np.random.uniform(-1, 1, (num_points, 4))),
        opacities=jnp.array(np.random.uniform(0, 1, (num_points, 1)))
    )
    # Normalize quats
    quats = gaussians.quaternions / jnp.linalg.norm(gaussians.quaternions, axis=-1, keepdims=True)
    gaussians = gaussians.replace(quaternions=quats)
    
    # Save
    ply_path = os.path.join(tmp_path, "test.ply")
    save_ply(ply_path, gaussians)
    
    # Load
    loaded_gaussians = load_ply(ply_path)
    
    # Verify
    np.testing.assert_allclose(gaussians.means, loaded_gaussians.means, atol=1e-5)
    np.testing.assert_allclose(gaussians.scales, loaded_gaussians.scales, atol=1e-5)
    np.testing.assert_allclose(gaussians.quaternions, loaded_gaussians.quaternions, atol=1e-5)
    np.testing.assert_allclose(gaussians.opacities, loaded_gaussians.opacities, atol=1e-5)
    
    # Verify SH (DC term only for simplicity in check, but logic matches)
    np.testing.assert_allclose(gaussians.sh_coeffs, loaded_gaussians.sh_coeffs, atol=1e-5)

def test_ply_loading_bad_path():
    with pytest.raises(FileNotFoundError):
        load_ply("non_existent_file.ply")
