import os
import h5py
import json
import concurrent.futures
from threading import Thread
from collections import OrderedDict
import numpy as np
from typing import Dict, List, Any, Tuple

from unav.core.colmap.read_write_model import read_model

# ==============================================================================
# LRU Feature Cache
# ==============================================================================

class LRUFeatureCache:
    """
    Fixed-size Least Recently Used (LRU) cache for feature dictionaries.
    Automatically evicts the least recently used items when full.
    """
    def __init__(self, maxsize: int = 256):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: str):
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)  # Remove least recently used

    def __contains__(self, key: str):
        return key in self.cache

    def __len__(self):
        return len(self.cache)

# Global cache instance for local feature blocks
feature_cache = LRUFeatureCache(maxsize=2560)

# ==============================================================================
# COLMAP Model Loader
# ==============================================================================

def load_colmap_model(
    model_dir: str,
    ext: str = "bin"
) -> Dict[str, Dict[str, Any]]:
    """
    Load a COLMAP sparse model and create a frame-centric dictionary for localization.

    Args:
        model_dir (str): Directory containing the COLMAP model.
        ext (str): Model file extension ('bin' or 'txt').

    Returns:
        Dict[str, Dict[str, Any]]: Mapping from image name to:
            - 'qvec' (np.ndarray): Camera quaternion (w, x, y, z).
            - 'tvec' (np.ndarray): Camera translation vector (x, y, z).
            - 'points2D_xy' (np.ndarray): 2D keypoint coordinates [N, 2].
            - 'points3D_xyz' (List[np.ndarray|None]): 3D coordinates for each keypoint (None if missing).
            - 'points3D_id' (np.ndarray): 3D point ID for each keypoint (-1 if none).
    """
    cameras, images, points3D = read_model(model_dir, ext)
    frames_by_name: Dict[str, Dict[str, Any]] = {}
    for img in images.values():
        pts2d = img.xys                      # 2D keypoints: shape [N, 2]
        pts3d_ids = img.point3D_ids          # 3D point IDs: shape [N]
        pts3d_xyz = [
            points3D[pid].xyz if pid != -1 and pid in points3D else None
            for pid in pts3d_ids
        ]
        frames_by_name[img.name] = {
            "qvec": img.qvec,
            "tvec": img.tvec,
            "points2D_xy": pts2d,
            "points3D_xyz": pts3d_xyz,
            "points3D_id": pts3d_ids,
        }
    return frames_by_name

# ==============================================================================
# Global Feature Loader
# ==============================================================================

def load_global_features(
    h5_path: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Load global image descriptors and names from an HDF5 file.

    Args:
        h5_path (str): Path to the global features HDF5.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - features: Array [N, D], where N = number of images, D = descriptor size.
            - image_names: List of image names (same order as features).
    """
    with h5py.File(h5_path, 'r') as f:
        image_names = list(f.keys())
        features = [f[name][()] for name in image_names]
    features = np.stack(features, axis=0)
    return features, image_names

# ==============================================================================
# Local Feature Loader with Caching and Background Prefetch
# ==============================================================================

def load_local_features(
    h5_path: str,
    image_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Load local features for specified images, using LRU cache and background prefetch.

    Args:
        h5_path (str): Path to local features HDF5 file.
        image_names (List[str]): Image names to load.

    Returns:
        Dict[str, Dict[str, Any]]: image_name -> feature dict with:
            - 'keypoints': np.ndarray [N, 2]
            - 'descriptors': np.ndarray [N, D]
            - 'scores': np.ndarray [N]
            - 'image_size': np.ndarray [2]
    """
    result = {}
    n_threads = os.cpu_count() or 4

    # Step 1: Load requested features, using cache if available
    to_load = [name for name in image_names if name not in feature_cache]
    if to_load:
        with h5py.File(h5_path, 'r') as f:
            def read_one(name):
                group = f[name]
                return name, {
                    "keypoints": group["keypoints"][()],
                    "descriptors": group["descriptors"][()],
                    "scores": group["scores"][()],
                    "image_size": group["image_size"][()],
                }
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                for name, feat in executor.map(read_one, to_load):
                    feature_cache.set(name, feat)

    # Assemble results from cache
    for name in image_names:
        result[name] = feature_cache.get(name)

    # Step 2: Launch background prefetch for spatially-nearby features
    Thread(target=background_preload, args=(h5_path, image_names)).start()

    return result

# ==============================================================================
# Background Feature Prefetching
# ==============================================================================

def background_preload(
    h5_path: str,
    center_names: List[str],
    window: int = 10
):
    """
    Background prefetching of features near current requests for cache warming.

    Args:
        h5_path (str): Path to local features HDF5.
        center_names (List[str]): Center image names for prefetch window.
        window (int): Number of neighbors to prefetch before and after each center.
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            all_names = list(f.keys())
        neighbor_set = set()
        for name in center_names:
            if name in all_names:
                idx = all_names.index(name)
                for offset in range(-window, window + 1):
                    ni = idx + offset
                    if 0 <= ni < len(all_names):
                        neighbor_set.add(all_names[ni])
        to_preload = [n for n in neighbor_set if n not in feature_cache]
        if to_preload:
            with h5py.File(h5_path, 'r') as f:
                def read_one(name):
                    group = f[name]
                    return name, {
                        "keypoints": group["keypoints"][()],
                        "descriptors": group["descriptors"][()],
                        "scores": group["scores"][()],
                        "image_size": group["image_size"][()],
                    }
                n_threads = min(os.cpu_count() or 4, 8)
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                    for name, feat in executor.map(read_one, to_preload):
                        feature_cache.set(name, feat)
    except Exception as e:
        print("[Background Prefetch] Error:", e)

# ==============================================================================
# Transformation Matrix Loader
# ==============================================================================

def load_transformation_matrix(
    file_path: str
) -> np.ndarray:
    """
    Load a transformation matrix (e.g., floorplan-to-SLAM) from a text or JSON file.

    Args:
        file_path (str): Path to the matrix file (.txt or .json supported).

    Returns:
        np.ndarray: Transformation matrix of shape [3, 3] or [4, 4].

    Raises:
        ValueError: If the file cannot be parsed.
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        matrix = np.array(data["transformation_matrix"])
    else:
        # Supports whitespace or comma-separated values
        matrix = np.loadtxt(file_path, delimiter=None)
    return matrix
