
import numpy as np
import struct

def save_ply(path, gaussians):
    """
    Save Gaussians to a PLY file compatible with 3DGS viewers (like Viser).
    Args:
        path (str): Output path.
        gaussians (Gaussians): JAX Gaussians namedtuple/dataclass.
    """
    print(f"Saving PLY to {path}...")
    
    xyz = np.array(gaussians.means)
    normals = np.zeros_like(xyz)
    
    # SH: (N, 16, 3)
    # Flatten to (N, 48) -> f_dc (3) + f_rest (45)
    # Convention: f_dc_0, f_dc_1, f_dc_2, f_rest_0...
    
    sh = np.array(gaussians.sh_coeffs)
    # Transpose to match PLY convention if needed?
    # Original: f_dc_0 (R), f_dc_1 (G), f_dc_2 (B)
    f_dc = sh[:, 0, :].reshape(-1, 3)
    f_rest = sh[:, 1:, :].reshape(-1, 45)
    
    opacities = np.array(gaussians.opacities)
    scales = np.array(gaussians.scales)
    quats = np.array(gaussians.quaternions)
    
    # Construct structured array
    # Define dtype
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
    
    data['x'] = xyz[:, 0]
    data['y'] = xyz[:, 1]
    data['z'] = xyz[:, 2]
    data['nx'] = normals[:, 0]
    data['ny'] = normals[:, 1]
    data['nz'] = normals[:, 2]
    data['f_dc_0'] = f_dc[:, 0]
    data['f_dc_1'] = f_dc[:, 1]
    data['f_dc_2'] = f_dc[:, 2]
    
    for i in range(45):
        data[f'f_rest_{i}'] = f_rest[:, i]
        
    data['opacity'] = opacities[:, 0]
    data['scale_0'] = scales[:, 0]
    data['scale_1'] = scales[:, 1]
    data['scale_2'] = scales[:, 2]
    data['rot_0'] = quats[:, 0]
    data['rot_1'] = quats[:, 1]
    data['rot_2'] = quats[:, 2]
    data['rot_3'] = quats[:, 3]
    
    # Write header
    with open(path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {num_points}\n".encode())
        
        for name, _ in dtype:
            f.write(f"property float {name}\n".encode())
            
        f.write(b"end_header\n")
        
        # Write binary data
        f.write(data.tobytes())
    
    print("Done.")
