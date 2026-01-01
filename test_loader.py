
from utils import load_colmap_data
import os

def test_loading():
    base_path = "/Users/mghifary/Work/Code/AI/jax-gs/data/nerf_example_data/nerf_llff_data/fern"
    xyz, rgb, cameras = load_colmap_data(base_path, images_dir_name="images_4")
    
    print(f"Total points: {len(xyz)}")
    print(f"Total cameras: {len(cameras)}")
    
    if len(cameras) > 0:
        cam = cameras[0]
        print(f"Camera 0 Image Shape: {cam.image.shape}")
        print(f"Camera 0 Model Resolution: {cam.width}x{cam.height}")
        
        # check if resizing is needed
        if cam.image.shape[1] != cam.width:
            print(f"MISMATCH: Image width {cam.image.shape[1]} != Model width {cam.width}")
            ratio = cam.width / cam.image.shape[1]
            print(f"Ratio: {ratio}")

if __name__ == "__main__":
    test_loading()
