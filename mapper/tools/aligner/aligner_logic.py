import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from mapper.tools.aligner.computation import compute_transform_matrix
from mapper.tools.aligner.io_utils import (
    load_temp_correspondences, save_temp_correspondences, save_matrix, load_matrix
)

class AlignerLogic:
    """
    Logic handler for keyframe-to-floorplan registration.
    Manages correspondences, transformation matrix computation, and I/O operations.
    No GUI logic is included in this class.
    """

    def __init__(
        self,
        kf_data: Dict[str, Any],
        keyframe_dir: str,
        floorplan_path: str,
        scale: float,
        config: Dict[str, Any]
    ) -> None:
        """
        Initialize AlignerLogic with all necessary paths and metadata.

        Args:
            kf_data (dict): Keyframe metadata dictionary.
            keyframe_dir (str): Directory containing keyframe images.
            floorplan_path (str): Path to the floorplan image.
            scale (float): Metric scale (e.g., meters per pixel).
            config (dict): Configuration dictionary for aligner, including temp/final dir paths.
        """
        self.kf_data = kf_data
        self.keyframe_dir = keyframe_dir
        self.floorplan_path = floorplan_path
        self.scale = scale
        self.config = config
        self.temp_dir = config['temp_dir']
        self.final_dir = config['final_dir']

        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)

        self.correspondences: List[Dict[str, Any]] = self._load_or_init_state()
        self.transform_matrix: Optional[np.ndarray] = self._load_matrix()

    def _load_or_init_state(self) -> List[Dict[str, Any]]:
        """
        Load correspondences from disk, or return an empty list if unavailable.

        Returns:
            List of correspondence dictionaries.
        """
        try:
            return load_temp_correspondences(self.temp_dir)
        except Exception:
            return []

    def save_to_matrix(self, matrix: Optional[np.ndarray] = None) -> None:
        """
        Save the transformation matrix to the final directory.
        If matrix is None, save self.transform_matrix.

        Args:
            matrix (np.ndarray, optional): Transformation matrix to save.
        """
        if matrix is None:
            matrix = self.transform_matrix
        if matrix is not None:
            save_matrix(self.final_dir, matrix)

    def _load_matrix(self) -> Optional[np.ndarray]:
        """
        Load the transformation matrix from the final directory, if it exists.

        Returns:
            Loaded matrix as np.ndarray or None.
        """
        try:
            return load_matrix(self.final_dir)
        except Exception:
            return None

    def save_correspondences(self) -> None:
        """
        Save all current correspondences to the temporary directory.
        """
        save_temp_correspondences(self.temp_dir, self.correspondences)

    def add_or_update_correspondence(self, corr: Dict[str, Any]) -> None:
        """
        Add a new correspondence or update an existing one by keyframe and keypoint index.

        Args:
            corr (dict): The correspondence to add or update.
        """
        keyframe = corr["keyframe"]
        keypoint_idx = corr["keypoint_idx"]
        updated = False
        for entry in self.correspondences:
            if entry["keyframe"] == keyframe and entry["keypoint_idx"] == keypoint_idx:
                entry.update(corr)
                updated = True
                break
        if not updated:
            self.correspondences.append(corr)
        self.save_correspondences()

    def update_last_floorplan(self, corr: Dict[str, Any]) -> None:
        """
        Update the floorplan 2D coordinates of the last correspondence.

        Args:
            corr (dict): The new correspondence containing updated 2D floorplan coordinates.
        """
        if self.correspondences and self.correspondences[-1] is corr:
            self.correspondences[-1]["floor2d"] = corr["floor2d"]
            self.save_correspondences()

    def get_correspondence_points(self) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Get all completed 2D-3D pairs where both floor2d and point3d are set.

        Returns:
            (points2D, points3D): Two lists of equal length.
        """
        completed = [
            c for c in self.correspondences
            if c.get("floor2d") is not None and c.get("point3d") is not None
        ]
        points2D = [c["floor2d"] for c in completed]
        points3D = [c["point3d"] for c in completed]
        return points2D, points3D

    def compute_transform(self) -> np.ndarray:
        """
        Compute and cache the transformation matrix from the current correspondences.

        Returns:
            The computed transformation matrix as np.ndarray.
        """
        points2D, points3D = self.get_correspondence_points()
        self.transform_matrix = compute_transform_matrix(points2D, points3D)
        return self.transform_matrix

    def get_landmarks_3d(self) -> np.ndarray:
        """
        Get all unique 3D landmarks observed by any keyframe.

        Returns:
            (N, 3) numpy array of unique 3D points.
        """
        all_pts = []
        for img_data in self.kf_data.values():
            if "matched_3d" in img_data:
                all_pts.extend(tuple(map(tuple, img_data["matched_3d"])))
        unique_pts = np.array(list(set(all_pts)), dtype=np.float32) if all_pts else np.zeros((0, 3), dtype=np.float32)
        return unique_pts

    def get_cam_trajectory_3d(self) -> np.ndarray:
        """
        Get the camera trajectory as an array of camera centers in world coordinates.

        Returns:
            (M, 3) numpy array.
        """
        traj = []
        for img_data in self.kf_data.values():
            if "T_cw" in img_data:
                T_cw = np.asarray(img_data["T_cw"])
                t = T_cw[:3, 3]
                traj.append(t)
        return np.array(traj, dtype=np.float32) if traj else np.zeros((0, 3), dtype=np.float32)

    def get_current_camera_pose(self, key: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Get the camera pose (center and rotation) for a specific keyframe.

        Args:
            key (str, optional): Keyframe image name.

        Returns:
            dict: {"center": t, "R": R}
        """
        if not self.kf_data:
            return {"center": np.zeros(3), "R": np.eye(3, dtype=np.float32)}
        if key is None:
            key = list(self.kf_data.keys())[-1]
        img_data = self.kf_data[key]
        T_cw = np.asarray(img_data["T_cw"])
        R = T_cw[:3, :3]
        t = T_cw[:3, 3]
        return {"center": t, "R": R}

    def get_current_observed_points(self, key: Optional[str] = None) -> np.ndarray:
        """
        Get all 3D points observed by the specified keyframe.

        Args:
            key (str, optional): Keyframe image name.

        Returns:
            (K, 3) numpy array.
        """
        if not self.kf_data:
            return np.zeros((0, 3), dtype=np.float32)
        if key is None:
            key = list(self.kf_data.keys())[-1]
        img_data = self.kf_data[key]
        if "matched_3d" in img_data and img_data["matched_3d"]:
            return np.array(img_data["matched_3d"], dtype=np.float32)
        else:
            return np.zeros((0, 3), dtype=np.float32)
