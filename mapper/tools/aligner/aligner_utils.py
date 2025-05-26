import re
import numpy as np
from PIL import Image
from typing import Any, List, Tuple

def natural_sort_key(s: str) -> List[Any]:
    """Generate a natural sort key for filenames or similar strings."""
    parts = re.split(r"(\d+)", s)
    return [int(text) if text.isdigit() else text.lower() for text in parts]

def compute_floorplan_display_size(floorplan_path: str, max_display=(1024, 768)) -> Tuple[int, int]:
    """Return the best display size for the floorplan image, keeping aspect ratio."""
    img = Image.open(floorplan_path)
    img.thumbnail(max_display, Image.Resampling.LANCZOS)
    return img.size

def compute_arrow_length(cam_traj_3d, ratio=0.08, min_len=0.01, max_len=100.0):
    if cam_traj_3d is not None and len(cam_traj_3d) > 1:
        traj_pts = np.asarray(cam_traj_3d)
        scene_range = np.ptp(traj_pts, axis=0)
        arrow_len = max(scene_range) * ratio
        return np.clip(arrow_len, min_len, max_len)
    else:
        return 0.08  # fallback value

