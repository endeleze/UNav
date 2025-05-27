import re
import numpy as np
from PIL import Image
from typing import Any, List, Tuple

def natural_sort_key(s: str) -> List[Any]:
    """
    Generate a natural sort key for filenames or similar strings,
    ensuring numerical parts are sorted as numbers, not text.

    Args:
        s (str): Input string.

    Returns:
        List[Any]: List for use as a sort key.
    """
    parts = re.split(r"(\d+)", s)
    return [int(text) if text.isdigit() else text.lower() for text in parts]

def compute_floorplan_display_size(
    floorplan_path: str,
    max_display: Tuple[int, int] = (1024, 768)
) -> Tuple[int, int]:
    """
    Compute the best display size for a floorplan image, preserving aspect ratio.

    Args:
        floorplan_path (str): Path to the floorplan image.
        max_display (Tuple[int, int]): Maximum width and height.

    Returns:
        Tuple[int, int]: Display width and height.
    """
    img = Image.open(floorplan_path)
    img.thumbnail(max_display, Image.Resampling.LANCZOS)
    return img.size

def compute_arrow_length(
    cam_traj_3d: np.ndarray,
    ratio: float = 0.08,
    min_len: float = 0.01,
    max_len: float = 100.0
) -> float:
    """
    Compute a reasonable arrow length for visualizing camera trajectories.

    Args:
        cam_traj_3d (np.ndarray): (N, 3) array of trajectory points.
        ratio (float): Ratio of scene span to use as arrow length.
        min_len (float): Minimum allowed arrow length.
        max_len (float): Maximum allowed arrow length.

    Returns:
        float: Arrow length.
    """
    if cam_traj_3d is not None and len(cam_traj_3d) > 1:
        traj_pts = np.asarray(cam_traj_3d)
        scene_range = np.ptp(traj_pts, axis=0)
        arrow_len = max(scene_range) * ratio
        return float(np.clip(arrow_len, min_len, max_len))
    else:
        return 0.08  # Fallback value for insufficient trajectory data
