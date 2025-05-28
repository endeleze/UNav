import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

def save_matrix(save_dir: str, matrix: np.ndarray) -> None:
    """
    Save the transformation matrix to disk as a .npy file.

    Args:
        save_dir (str): Directory in which to save the matrix.
        matrix (np.ndarray): Transformation matrix to save.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "transform_matrix.npy")
    np.save(save_path, matrix)
    print(f"[✓] Transform matrix saved to {save_path}")

def load_matrix(load_dir: str) -> np.ndarray:
    """
    Load the transformation matrix from disk.

    Args:
        load_dir (str): Directory from which to load the matrix.

    Returns:
        np.ndarray: The loaded transformation matrix.

    Raises:
        FileNotFoundError: If the matrix file does not exist.
        Exception: For any other I/O or deserialization error.
    """
    load_path = os.path.join(load_dir, "transform_matrix.npy")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No transform matrix found at {load_path}")
    matrix = np.load(load_path)
    print(f"[✓] Transform matrix loaded from {load_path}")
    return matrix

def load_scales(scale_file: Optional[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load a hierarchical JSON file containing metric scale values for each floor.

    The JSON should have the structure:
        {
            "place": {
                "building": {
                    "floor": scale_value
                }
            }
        }

    Args:
        scale_file (str or None): Path to the JSON file.

    Returns:
        dict: Nested dictionary {place: {building: {floor: scale}}}
    """
    scales = {}
    if scale_file and os.path.exists(scale_file):
        with open(scale_file, 'r') as f:
            data = json.load(f)
        for place, buildings in data.items():
            scales.setdefault(place, {})
            for bld, floors in buildings.items():
                scales[place].setdefault(bld, {})
                for fl, sc in floors.items():
                    scales[place][bld][fl] = sc
    return scales

def save_temp_correspondences(temp_dir: str, correspondences: List[Dict[str, Any]]) -> None:
    """
    Save the current list of correspondences to disk for GUI recovery,
    handling numpy types for JSON compatibility.

    Args:
        temp_dir (str): Directory to save the temporary JSON file.
        correspondences (List[Dict[str, Any]]): Correspondence list.
    """
    os.makedirs(temp_dir, exist_ok=True)
    save_path = os.path.join(temp_dir, "correspondences.json")

    def _default_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic,)):
            return obj.item()
        raise TypeError(f"Type {obj.__class__.__name__} not serializable")

    with open(save_path, "w") as f:
        json.dump(correspondences, f, indent=2, default=_default_serializer)

def load_temp_correspondences(temp_dir: str) -> List[Dict[str, Any]]:
    """
    Load the list of correspondences saved for GUI recovery.

    Args:
        temp_dir (str): Directory containing 'correspondences.json'.

    Returns:
        List[Dict[str, Any]]: List of correspondence records, or [] if none found.
    """
    path = os.path.join(temp_dir, "correspondences.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_final_matrix(output_dir: str, matrix: np.ndarray) -> None:
    """
    Save the final 3D-to-2D transformation matrix after alignment as a .npy file.

    Args:
        output_dir (str): Directory to save the final matrix.
        matrix (np.ndarray): The transformation matrix (typically 2x4 or 3x4).
    """
    os.makedirs(output_dir, exist_ok=True)
    matrix_path = os.path.join(output_dir, "transform_matrix.npy")
    np.save(matrix_path, matrix)
    print(f"[✓] Final transform matrix saved to {matrix_path}")
