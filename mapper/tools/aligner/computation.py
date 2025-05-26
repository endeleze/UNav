import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def compute_transform_matrix(
    points2D: List[Tuple[float, float]],
    points3D: List[List[float]]
) -> np.ndarray:
    """
    Compute the 2D affine transformation matrix that maps 3D points to 2D points via least squares.

    Args:
        points2D (List[Tuple[float, float]]): List of 2D coordinates on the floorplan (shape: N x 2).
        points3D (List[List[float]]): List of corresponding 3D coordinates (shape: N x 3).

    Returns:
        np.ndarray: Affine transformation matrix of shape (2, 4), such that:
                    [u, v]^T ≈ T @ [X, Y, Z, 1]^T

    Raises:
        ValueError: If input lengths do not match or if fewer than 4 correspondences are provided.
    """
    if len(points2D) != len(points3D):
        raise ValueError(f"Mismatched number of 2D and 3D points: {len(points2D)} vs {len(points3D)}")
    if len(points2D) < 4:
        raise ValueError("At least 4 point correspondences are required to compute the affine transform.")

    # Convert input lists to numpy arrays
    pts2D = np.array(points2D, dtype=np.float32)  # (N, 2)
    pts3D = np.array(points3D, dtype=np.float32)  # (N, 3)

    # Convert 3D points to homogeneous coordinates (N, 4)
    pts3D_h = np.concatenate([pts3D, np.ones((pts3D.shape[0], 1), dtype=np.float32)], axis=1)

    # Solve the least squares problem: find T (4, 2) such that pts3D_h @ T ≈ pts2D
    T, _, _, _ = np.linalg.lstsq(pts3D_h, pts2D, rcond=None)  # T: (4, 2)
    T = T.T  # Transpose to shape (2, 4)
    return T

def project_point3d_to_floor2d(point3d: List[float], T_3d_to_2d: np.ndarray) -> List[float]:
    """
    Projects a single 3D point to 2D using the provided affine transformation matrix.

    Args:
        point3d (List[float]): The 3D point [X, Y, Z].
        T_3d_to_2d (np.ndarray): The (2, 4) affine transformation matrix.

    Returns:
        List[float]: The projected 2D point [u, v].
    """
    pt3d = np.asarray(point3d).reshape(1, 3)
    ones = np.ones((1, 1))
    pt3d_h = np.hstack([pt3d, ones])  # (1, 4)
    pt2d = (T_3d_to_2d @ pt3d_h.T).T[0]  # shape: (2,)
    return pt2d.tolist()

def lists_equal(l1: Any, l2: Any) -> bool:
    """
    Checks if two lists (which may be nested and/or contain numpy arrays) are element-wise equal.

    Args:
        l1, l2: Lists, tuples, or numpy arrays to compare.

    Returns:
        bool: True if the lists are element-wise equal, False otherwise.
    """
    if l1 is l2:
        return True
    if l1 is None or l2 is None:
        return (l1 in (None, [], ())) and (l2 in (None, [], ()))
    if len(l1) != len(l2):
        return False
    for a, b in zip(l1, l2):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if not np.array_equal(a, b):
                return False
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if not lists_equal(a, b):
                return False
        else:
            if a != b:
                return False
    return True

def pose_dicts_equal(d1: Optional[Dict], d2: Optional[Dict], atol: float = 1e-8) -> bool:
    """
    Checks if two dictionaries containing numpy arrays as values are element-wise close.

    Args:
        d1, d2 (dict or None): The dictionaries to compare.
        atol (float): Absolute tolerance for floating point comparisons.

    Returns:
        bool: True if the dicts have the same keys and all arrays are element-wise close.
    """
    if d1 is d2:
        return True
    if d1 is None or d2 is None:
        return d1 is d2
    if set(d1.keys()) != set(d2.keys()):
        return False
    for k in d1.keys():
        if not np.allclose(d1[k], d2[k], atol=atol):
            return False
    return True
