
import viser
import time
import argparse
import os
import numpy as np


# Since I don't know if 'plyfile' is installed, I'll use a pure numpy approach if possible 
# or implement a simple PLY reader for the specific format we saved.
# We saved: x,y,z, nx,ny,nz, f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3

def load_ply(path):
    # Mapping from our PLY format to Viser arguments
    # Viser expects: centers, covariances, rgbs, opacities
    
    # We need to manually parse the binary PLY because Viser's API specifically asks for arrays,
    # not a PLY file blob (the error message confirmed this: missing covariances, etc).
    
    # To properly load the custom PLY we made, we need to read it back.
    # Our save_utils.py used a specific struct.
    
    # We can use a simple custom parser for binary PLY.
    
    with open(path, "rb") as f:
        # Read header to find end
        line = ""
        header_end = False
        num_vertices = 0
        property_names = []
        
        while not header_end:
            line = f.readline().decode('utf-8').strip()
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property float"):
                property_names.append(line.split()[-1])
            elif line == "end_header":
                header_end = True
        
        # Read binary data
        # All properties are float32
        data = np.fromfile(f, dtype=np.float32)
        
    num_props = len(property_names)
    expected_size = num_vertices * num_props
    
    if data.size != expected_size:
        print(f"Error: Expected {expected_size} floats, got {data.size}")
        return None, None, None, None
        
    data = data.reshape(num_vertices, num_props)
    
    # Map properties to indices
    prop_map = {name: i for i, name in enumerate(property_names)}
    
    # Extract Means (x, y, z)
    means = data[:, [prop_map['x'], prop_map['y'], prop_map['z']]]
    
    # Extract Opacities
    opacities = data[:, prop_map['opacity']]
    
    # Extract Colors (SH DC) -> Convert to RGB
    # SH DC 0 is R, DC 1 is G, DC 2 is B (approx, actually need to add 0.5 and scale)
    # in save_utils: f_dc = (color - 0.5) / 0.282...
    # so color = f_dc * 0.282... + 0.5
    f_dc = data[:, [prop_map['f_dc_0'], prop_map['f_dc_1'], prop_map['f_dc_2']]]
    colors = f_dc * 0.28209479177387814 + 0.5
    colors = np.clip(colors, 0.0, 1.0)
    
    # Extract Scales
    scales_data = data[:, [prop_map['scale_0'], prop_map['scale_1'], prop_map['scale_2']]]
    scales = np.exp(scales_data) # We stored log scales? save_utils says "scales = np.array(gaussians.scales)". 
    # Init code says: scales = log distance. So yes, they are log scales.
    
    # Extract Rotations (Quaternions)
    quats = data[:, [prop_map['rot_0'], prop_map['rot_1'], prop_map['rot_2'], prop_map['rot_3']]]
    
    # Compute Covariances
    # We need to construct rotation matrices and scale matrices
    # R from quaternion
    # S from scale
    # Cov = R S S^T R^T
    
    # Normalized quats
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    q = quats / norms
    
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    R = np.zeros((num_vertices, 3, 3))
    
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
    S = np.zeros((num_vertices, 3, 3))
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]
    
    # M = R @ S
    M = np.einsum('nij,njk->nik', R, S)
    
    # Cov = M @ M.T
    covs = np.einsum('nij,nkj->nik', M, M)
    
    return means, covs, colors, opacities

def run_ply_viewer(ply_path):
    print(f"Parsing PLY from {ply_path}...")
    means, covs, colors, opacities = load_ply(ply_path)
    
    if means is None:
        return

    print(f"Loaded {len(means)} splats")
    
    server = viser.ViserServer()

    # Set dark mode
    server.configure_theme(dark_mode=True)
    
    
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        print(f"Client connected: {client.client_id}")
        
        client.scene.add_gaussian_splats(
            "/gaussians",
            centers=np.ascontiguousarray(means),
            covariances=np.ascontiguousarray(covs),
            rgbs=np.ascontiguousarray(colors),
            opacities=np.ascontiguousarray(opacities.reshape(-1, 1)), # Reshape to (N, 1) per assertion
            visible=True,
        )
        
        # Make it look nice
        # Try looking from +Z for OpenCV style (Y down)
        client.camera.position = (0.0, 0.0, -5.0) # Back up along Z
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.up_direction = (0.0, -1.0, 0.0) # Y is down in COLMAP/OpenCV

    print("Viewer running... Press Ctrl+C to stop.")
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_path", type=str, help="Path to the .ply file")
    args = parser.parse_args()
    
    if not os.path.exists(args.ply_path):
        print(f"Error: File {args.ply_path} not found.")
    else:
        run_ply_viewer(args.ply_path)
