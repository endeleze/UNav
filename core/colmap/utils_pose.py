import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

def load_colmap_images_file(image_file: str) -> dict:
    """
    Load COLMAP standard images.txt.
    Returns:
        dict: {image_name: 4x4 pose matrix}
    """
    poses = {}
    with open(image_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.strip() == '' or line.startswith('#'):
            continue
        tokens = line.strip().split()
        if len(tokens) < 10:
            continue  # Skip invalid lines
        qw, qx, qy, qz = map(float, tokens[1:5])
        tx, ty, tz = map(float, tokens[5:8])
        img_name = tokens[9]

        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array([tx, ty, tz])

        R_cw = rot
        t_cw = t

        # Inverse: T_wc = [R_wc | t_wc]
        # where R_wc = R_cw^T, t_wc = -R_cw^T @ t_cw
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw

        pose_mat = np.eye(4)
        pose_mat[:3, :3] = R_wc
        pose_mat[:3, 3] = t_wc
        poses[img_name] = pose_mat

    print(f"[✅] Loaded {len(poses)} poses from {image_file}")
    return poses

def load_colmap_images_file_qt(images_file: str) -> dict:
    """
    Load COLMAP standard images.txt.
    Returns:
        dict: {image_name: {'qvec': np.array, 'tvec': np.array, 'image_id': int}}
    """
    poses = {}
    with open(images_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.strip() == '' or line.startswith('#'):
            continue
        tokens = line.strip().split()
        if len(tokens) < 10:
            continue  # Skip invalid lines

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