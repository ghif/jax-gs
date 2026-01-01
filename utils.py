
import struct
import numpy as np
import jax.numpy as jnp
from PIL import Image
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class CameraInfo:
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int

@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            if model_id == 0: # SIMPLE_PINHOLE
                params = struct.unpack("<3d", fid.read(24))
                f = params[0]
                cx = params[1]
                cy = params[2]
                fx = f
                fy = f
            elif model_id == 1: # PINHOLE
                params = struct.unpack("<4d", fid.read(32))
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
            elif model_id == 2: # SIMPLE_RADIAL
                params = struct.unpack("<4d", fid.read(32))
                f = params[0]
                cx = params[1]
                cy = params[2]
                k = params[3]
                fx = f
                fy = f
            else:
                 # Minimal implementation, might need to expand if other models appear
                raise NotImplementedError(f"Camera model {model_id} not implemented")

            cameras[camera_id] = {
                "id": camera_id,
                "model": "PINHOLE",
                "width": width,
                "height": height,
                "params": (fx, fy, cx, cy)
            }
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            # i (4), 4d (32), 3d (24), i (4) = 64 bytes
            binary_image_properties = struct.unpack("<idddddddi", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]
                
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            fid.read(num_points2D * 24) # Skip points2D data
            
            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name
            }
    return images

def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            binary_point_properties = struct.unpack("<QdddBBBd", fid.read(43))
            point3D_id = binary_point_properties[0]
            xyz = np.array(binary_point_properties[1:4])
            rgb = np.array(binary_point_properties[4:7])
            error = binary_point_properties[7]
            track_length = struct.unpack("<Q", fid.read(8))[0]
            fid.read(track_length * 8) # Skip track info
            
            points3D[point3D_id] = {
                "id": point3D_id,
                "xyz": xyz,
                "rgb": rgb,
                "error": error
            }
    return points3D

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def load_colmap_data(source_path, images_dir_name="images_4"):
    """
    Load COLMAP data from a directory.
    
    Args:
        source_path (str): Path to the COLMAP data directory.
        images_dir_name (str): Name of the images directory (default: "images_4").
    
    Returns:
        tuple: A tuple containing the following:
            - xyz (numpy.ndarray): Array of 3D points.
            - rgb (numpy.ndarray): Array of RGB colors.
            - train_cam_infos (list): List of CameraInfo objects.   
    """
    cameras_extrinsic_file = os.path.join(source_path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(source_path, "sparse/0", "cameras.bin")
    points3D_file = os.path.join(source_path, "sparse/0", "points3D.bin")
    
    print("Loading COLMAP data...")
    cam_extrinsics = read_images_binary(cameras_extrinsic_file)
    cam_intrinsics = read_cameras_binary(cameras_intrinsic_file)
    points3D_data = read_points3D_binary(points3D_file)
    
    # Process Points 3D
    xyz = np.array([p["xyz"] for p in points3D_data.values()])
    rgb = np.array([p["rgb"] for p in points3D_data.values()]) / 255.0
    
    print(f"Loaded {len(xyz)} 3D points")
    
    # Process Cameras
    train_cameras = []
    
    # We sort by image id to keep some order
    sorted_image_ids = sorted(cam_extrinsics.keys())

    # Check for image files
    images_dir = os.path.join(source_path, images_dir_name)
    if not os.path.exists(images_dir):
        print(f"Error: Images directory {images_dir} not found")
        return xyz, rgb, []

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Analyze if we need mapping
    sparse_names = [cam_extrinsics[k]["name"] for k in sorted_image_ids]
    use_index_mapping = False
    
    # Check if first image exists
    first_path = os.path.join(images_dir, sparse_names[0])
    if not os.path.exists(first_path):
        print(f"Exact filename {sparse_names[0]} not found in {images_dir}. Checking for index-based mapping...")
        if len(sparse_names) == len(image_files):
            print(f"Found {len(image_files)} images, same as model. using 1-to-1 mapping by sort order.")
            use_index_mapping = True
        else:
            print(f"Warning: Model has {len(sparse_names)} images, disk has {len(image_files)}. Mapping might be wrong.")
            # Still try to map if counts differ? Maybe just valid subset? 
            # If disk has MORE, we might be fine if we assume sorted, but risky.
            # If disk has LESS, definitely bad.
            # Let's try 1-to-1 for shared count?
            # For now, if sizes mismatch and names mismatch, we are in trouble.
            # But we'll try to use index mapping if requested essentially.
            if len(image_files) > 0:
                print("Attempting to use sorted list from disk.")
                use_index_mapping = True
    
    for i, image_id in enumerate(sorted_image_ids):
        extr = cam_extrinsics[image_id]
        intr = cam_intrinsics[extr["camera_id"]]
        
        height = intr["height"]
        width = intr["width"]
        
        R = qvec2rotmat(extr["qvec"])
        T = extr["tvec"]
        
        # Convert to 4x4 matrix
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        
        
        # FOV calculations
        # Params are always (fx, fy, cx, cy) now
        focal_length_x = intr["params"][0]
        focal_length_y = intr["params"][1]

        fovx = 2 * np.arctan(width / (2 * focal_length_x))
        fovy = 2 * np.arctan(height / (2 * focal_length_y))
        
        if use_index_mapping and i < len(image_files):
            image_name = image_files[i]
        else:
            image_name = extr["name"]
            
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found")
            continue
            
        # Load image
        image_pil = Image.open(image_path)
        image = np.array(image_pil)
        
        # Resize if needed (naive resize if resolution doesn't match intrinsics)
        # But usually images_4 matches the sparse model if run correctly
        # Here we just check and print warning
        if image.shape[1] != width or image.shape[0] != height:
             # If mismatch, assume we need to update intrinsics to match the loaded image
             scale_x = image.shape[1] / width
             scale_y = image.shape[0] / height
             
             # Update intrinsics
             focal_length_x *= scale_x
             focal_length_y *= scale_y
             fovx = 2 * np.arctan(image.shape[1] / (2 * focal_length_x))
             fovy = 2 * np.arctan(image.shape[0] / (2 * focal_length_y))
             
             width = image.shape[1]
             height = image.shape[0]
        
        image = image / 255.0
        
        train_cameras.append(CameraInfo(
            uid=image_id,
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height
        ))
        
    print(f"Loaded {len(train_cameras)} cameras")
    
    return xyz, rgb, train_cameras
