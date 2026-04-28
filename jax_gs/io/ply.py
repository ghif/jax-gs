import numpy as np
import jax.numpy as jnp
import fsspec
from jax_gs.core.gaussians import Gaussians
from jax_gs.core.gaussians_2d import Gaussians2D

def save_ply(path, gaussians: Gaussians):
    """
    Save Gaussians to a PLY file compatible with 3DGS viewers (like Viser).

    Args:
        path: Path to save the PLY file
        gaussians: Gaussians dataclass
    """
    print(f"Saving PLY to {path}...")
    
    xyz = np.array(gaussians.means)
    normals = np.zeros_like(xyz)
    
    sh = np.array(gaussians.sh_coeffs)
    f_dc = sh[:, 0, :].reshape(-1, 3)
    f_rest = sh[:, 1:, :].reshape(-1, 45)
    
    opacities = np.array(gaussians.opacities)
    scales = np.array(gaussians.scales)
    quats = np.array(gaussians.quaternions)
    
    # Define dtype for PLY properties
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')]
    
    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))
        
    dtype.append(('opacity', 'f4'))
    dtype.append(('scale_0', 'f4'))
    dtype.append(('scale_1', 'f4'))
    dtype.append(('scale_2', 'f4'))
    dtype.append(('rot_0', 'f4'))
    dtype.append(('rot_1', 'f4'))
    dtype.append(('rot_2', 'f4'))
    dtype.append(('rot_3', 'f4'))
    
    num_points = xyz.shape[0]
    data = np.zeros(num_points, dtype=dtype)
    
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['f_dc_0'], data['f_dc_1'], data['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    
    for i in range(45):
        data[f'f_rest_{i}'] = f_rest[:, i]
        
    data['opacity'] = opacities[:, 0]
    data['scale_0'], data['scale_1'], data['scale_2'] = scales[:, 0], scales[:, 1], scales[:, 2]
    data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3'] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    
    with fsspec.open(path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {num_points}\n".encode())
        for name, _ in dtype:
            f.write(f"property float {name}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())
    
    print("Done.")

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

def load_ply(path):
    """
    Load Gaussians from a PLY file.

    Args:
        path: Path to the PLY file
    Returns:
        gaussians: Gaussians dataclass
    """
    print(f"Loading PLY from {path}...")
    
    # Define the same dtype as in save_ply
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')]
    
    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))
        
    dtype.append(('opacity', 'f4'))
    dtype.append(('scale_0', 'f4'))
    dtype.append(('scale_1', 'f4'))
    dtype.append(('scale_2', 'f4'))
    dtype.append(('rot_0', 'f4'))
    dtype.append(('rot_1', 'f4'))
    dtype.append(('rot_2', 'f4'))
    dtype.append(('rot_3', 'f4'))
    
    with fsspec.open(path, 'rb') as f:
        line = f.readline()
        num_points = 0
        while line and line.strip() != b"end_header":
            if b"element vertex" in line:
                num_points = int(line.split()[-1])
            line = f.readline()
            
        data = np.fromfile(f, dtype=dtype, count=num_points)
        
    means = jnp.stack([data['x'], data['y'], data['z']], axis=-1)
    
    f_dc = jnp.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], axis=-1)
    f_rest = jnp.stack([data[f'f_rest_{i}'] for i in range(45)], axis=-1)
    sh_coeffs = jnp.concatenate([f_dc, f_rest], axis=-1).reshape(-1, 16, 3)
    
    opacities = data['opacity'][:, None]
    scales = jnp.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=-1)
    quats = jnp.stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']], axis=-1)
    
    print("Done.")
    return Gaussians(
        means=means,
        scales=scales,
        quaternions=quats,
        opacities=opacities,
        sh_coeffs=sh_coeffs
    )

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
