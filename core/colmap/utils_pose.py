import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

def load_colmap_images_file(image_file: str) -> dict:
    """
    Load COLMAP images.txt file and convert all poses to 4x4 world-to-camera matrices.

    Args:
        image_file (str): Path to images.txt.

    Returns:
        dict: Mapping {image_name: 4x4 np.ndarray, world-to-camera pose}
    """
    poses = {}
    with open(image_file, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            tokens = line.strip().split()
            if len(tokens) < 10:
                continue  # Skip incomplete or invalid lines
            # Parse quaternion and translation
            qw, qx, qy, qz = map(float, tokens[1:5])
            tx, ty, tz = map(float, tokens[5:8])
            img_name = tokens[9]
            # COLMAP format: QW QX QY QZ TX TY TZ
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()  # camera-to-world
            t = np.array([tx, ty, tz])

            # Compute world-to-camera transformation
            R_wc = rot.T
            t_wc = -R_wc @ t

            pose_mat = np.eye(4)
            pose_mat[:3, :3] = R_wc
            pose_mat[:3, 3] = t_wc
            poses[img_name] = pose_mat

    print(f"[✅] Loaded {len(poses)} poses from {image_file}")
    return poses

def load_colmap_images_file_qt(images_file: str) -> dict:
    """
    Load COLMAP images.txt file, returning quaternions and translations for each image.

    Args:
        images_file (str): Path to images.txt.

    Returns:
        dict: Mapping {image_name: {'qvec': np.array([qw, qx, qy, qz]), 'tvec': np.array([tx, ty, tz]), 'image_id': int}}
    """
    poses = {}
    with open(images_file, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            tokens = line.strip().split()
            if len(tokens) < 10:
                continue  # Skip incomplete or invalid lines

            image_id = int(tokens[0])
            qw, qx, qy, qz = map(float, tokens[1:5])
            tx, ty, tz = map(float, tokens[5:8])
            img_name = tokens[9]

            poses[img_name] = {
                'qvec': np.array([qw, qx, qy, qz], dtype=np.float64),
                'tvec': np.array([tx, ty, tz], dtype=np.float64),
                'image_id': image_id
            }

    logging.info(f"[UNav] Loaded {len(poses)} poses from {images_file}")
    return poses