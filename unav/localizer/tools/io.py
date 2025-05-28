import h5py
import numpy as np
from typing import Dict, List, Any, Tuple

from unav.core.colmap.read_write_model import read_model

def load_colmap_model(
    model_dir: str,
    ext: str = "bin"
) -> Dict[str, Dict[str, Any]]:
    """
    Load a COLMAP sparse model and build a frame-centric index for localization.

    Args:
        model_dir (str): Path to the COLMAP model directory.
        ext (str): File extension for COLMAP model files ('bin' or 'txt').

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping image name to a dictionary containing:
            - 'qvec' (np.ndarray): Camera rotation (quaternion, shape: [4,]).
            - 'tvec' (np.ndarray): Camera translation vector (shape: [3,]).
            - 'points2D_xy' (np.ndarray): 2D keypoint coordinates in the image (shape: [N, 2]).
            - 'points3D_xyz' (List[np.ndarray or None]): List of 3D coordinates for each keypoint (None if no 3D point).
            - 'points3D_id' (np.ndarray): List of 3D point IDs for each keypoint (-1 if not associated).
    """
    cameras, images, points3D = read_model(model_dir, ext)

    frames_by_name: Dict[str, Dict[str, Any]] = {}
    for img in images.values():
        pts2d = img.xys                      # Array of 2D keypoints (N, 2)
        pts3d_ids = img.point3D_ids          # Array of 3D point IDs (N,)
        pts3d_xyz = [
            points3D[pid].xyz if pid != -1 and pid in points3D else None
            for pid in pts3d_ids
        ]
        frames_by_name[img.name] = {
            "qvec": img.qvec,                # Quaternion [w, x, y, z]
            "tvec": img.tvec,                # Translation vector [x, y, z]
            "points2D_xy": pts2d,            # 2D coordinates of keypoints
            "points3D_xyz": pts3d_xyz,       # 3D coordinates for matched points (None if not available)
            "points3D_id": pts3d_ids         # 3D point IDs for each keypoint (-1 if none)
        }
    return frames_by_name


def load_global_features(
    h5_path: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Load all global descriptors and their image names from an HDF5 file.

    Args:
        h5_path (str): Path to the global features HDF5 file.

    Returns:
        Tuple[np.ndarray, List[str]]: (features, image_names)
            - features: Array of shape [N, D] (N images, D-dimensional descriptors)
            - image_names: List of image names, order matches the features array
    """
    with h5py.File(h5_path, 'r') as f:
        image_names = list(f.keys())
        features = [f[name][()] for name in image_names]
    features = np.stack(features, axis=0)
    return features, image_names


def load_local_features(
    h5_path: str,
    image_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Load local features for specified images from an HDF5 file.

    Args:
        h5_path (str): Path to the local features HDF5 file.
        image_names (List[str]): List of image names to load.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping image name to dict with keys:
            - 'keypoints' (np.ndarray): Keypoint coordinates [N, 2]
            - 'descriptors' (np.ndarray): Local descriptors [N, D]
            - 'scores' (np.ndarray): Keypoint scores [N]
            - 'image_size' (np.ndarray): [W, H]
    """
    result = {}
    with h5py.File(h5_path, 'r') as f:
        for name in image_names:
            group = f[name]
            result[name] = {
                "keypoints": group["keypoints"][()],
                "descriptors": group["descriptors"][()],
                "scores": group["scores"][()],
                "image_size": group["image_size"][()]
            }
    return result


def load_transformation_matrix(
    file_path: str
) -> np.ndarray:
    """
    Load a transformation matrix (floorplan-to-SLAM or similar) from a text or JSON file.

    Args:
        file_path (str): Path to the matrix file (supports .txt or .json).

    Returns:
        np.ndarray: The transformation matrix (shape: [3, 3] or [4, 4]).

    Raises:
        ValueError: If the file format is unsupported or the matrix cannot be parsed.
    """
    if file_path.endswith('.json'):
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        matrix = np.array(data["transformation_matrix"])
    else:
        # Assume whitespace- or comma-separated text
        matrix = np.loadtxt(file_path, delimiter=None)
    return matrix
