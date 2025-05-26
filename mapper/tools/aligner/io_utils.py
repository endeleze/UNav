import numpy as np
from typing import List, Dict, Any, Optional
import os
import json


def save_matrix(save_dir: str, matrix: np.ndarray):
    """
    Save the transformation matrix to disk.

    Args:
        save_dir (str): Directory or path to save.
        matrix (np.ndarray): Transformation matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "transform_matrix.npy")
    np.save(save_path, matrix)
    print(f"[✓] Transform matrix saved to {save_path}")

def load_matrix(load_dir: str) -> np.ndarray:
    """
    Load the transformation matrix from disk.

    Args:
        load_dir (str): Directory or path to load from.

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
    Load a hierarchical JSON or YAML file containing scale values.

    The hierarchy should be structured as:
    {
        "place": {
            "building": {
                "floor": scale_value
            }
        }
    }

    Args:
        scale_file: Path to the scale file.

    Returns:
        A nested dictionary containing scales at place/building/floor granularity.
    """
    scales = {}
    if scale_file and os.path.exists(scale_file):
        with open(scale_file, 'r') as f:
            data = json.load(f)
        
        for place, buildings in data.items():
            if place not in scales:
                scales[place] = {}
            for bld, floors in buildings.items():
                if bld not in scales[place]:
                    scales[place][bld] = {}
                for fl, sc in floors.items():
                    scales[place][bld][fl] = sc
    return scales

def save_temp_correspondences(temp_dir: str, correspondences: List[Dict[str, Any]]) -> None:
    """
    Save the entire correspondences list for recovery in GUI by converting any
    numpy arrays or scalars to native Python types for JSON serialization.

    Args:
        temp_dir (str): Directory to save the temporary JSON file.
        correspondences (List[Dict[str, Any]]): List of correspondence dictionaries.
    """
    os.makedirs(temp_dir, exist_ok=True)
    save_path = os.path.join(temp_dir, "correspondences.json")

    def _default_serializer(obj):
        # Convert numpy arrays to lists, numpy scalars to Python scalars
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic,)):
            return obj.item()
        # Let JSON encoder raise for any other unsupported type
        raise TypeError(f"Type {obj.__class__.__name__} not serializable")

    with open(save_path, "w") as f:
        json.dump(correspondences, f, indent=2, default=_default_serializer)
        
def load_temp_correspondences(temp_dir: str) -> List[dict]:
    """
    Load previously saved correspondences for recovery in GUI.

    Args:
        temp_dir (str): Path to directory containing 'correspondences.json'.

    Returns:
        List[dict]: List of correspondence dictionaries.
    """
    import json
    path = os.path.join(temp_dir, "correspondences.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_final_matrix(output_dir: str, matrix: np.ndarray) -> None:
    """
    Save final 3D-to-2D transformation matrix after manual alignment.

    Args:
        output_dir (str): Final directory to save the matrix.
        matrix (np.ndarray): 3x4 or 3x3 transformation matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    matrix_path = os.path.join(output_dir, "transform_matrix.npy")
    np.save(matrix_path, matrix)