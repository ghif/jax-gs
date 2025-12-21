import time
import viser
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R
from renderer import Camera, render
from gaussians import Gaussians, init_gaussians_from_pcd

def run_viewer():
    server = viser.ViserServer()
    
    # Initialize some dummy Gaussians for testing
    num_points = 10
    # Increase range and center them
    points = np.random.uniform(-2, 2, (num_points, 3))
    colors = np.random.uniform(0.5, 1.0, (num_points, 3)) # Brighter colors
    
    gaussians = init_gaussians_from_pcd(jnp.array(points), jnp.array(colors))
    # Override scales to be larger for visibility
    gaussians = gaussians.replace(scales=jnp.full((num_points, 3), -2.0)) # log(0.13)
    # Override opacities to be higher
    gaussians = gaussians.replace(opacities=jnp.full((num_points, 1), 2.0)) # sigmoid(2) ~ 0.88
    
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        print(f"Client connected: {client.client_id}")
        
        try:
            while True:
                # Get current camera pose from Viser
                c = client.camera
                
                # WXYZ to Rotation Matrix
                rot = R.from_quat([c.wxyz[1], c.wxyz[2], c.wxyz[3], c.wxyz[0]]).as_matrix()
                pos = np.array(c.position)
                
                # Create W2C matrix
                R_w2c = rot.T
                t_w2c = -R_w2c @ pos
                
                W2C = np.eye(4)
                W2C[:3, :3] = R_w2c
                W2C[:3, 3] = t_w2c
                
                # Adjust to GS coordinates
                flip_yz = np.diag([1, -1, -1, 1])
                W2C = flip_yz @ W2C
                
                # Camera setup - reduced resolution for CPU performance
                W, H = 192, 192 
                cam = Camera(
                    W=W, H=H,
                    fx=150.0, fy=150.0,
                    cx=W/2, cy=H/2,
                    W2C=jnp.array(W2C),
                    full_proj=jnp.eye(4)
                )
                
                image = render(gaussians, cam)
                jax.block_until_ready(image)
                
                # Send image to client
                client.scene.set_background_image(np.array(image), format="jpeg")
                
                time.sleep(0.01)
        except Exception as e:
            print(f"Connection error: {e}")

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    run_viewer()
