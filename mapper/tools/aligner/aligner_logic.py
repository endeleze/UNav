import os
import numpy as np

from mapper.tools.aligner.computation import compute_transform_matrix
from mapper.tools.aligner.io_utils import (
    load_temp_correspondences, save_temp_correspondences, save_matrix, load_matrix
)

class AlignerLogic:
    """
    Logic handler for keyframe-floorplan registration.
    Manages correspondences, transformation matrix, and IO operations.
    This class contains no GUI logic.
    """

    def __init__(self, kf_data, keyframe_dir, floorplan_path, scale, config):
        self.kf_data = kf_data
        self.keyframe_dir = keyframe_dir
        self.floorplan_path = floorplan_path
        self.scale = scale
        self.config = config
        self.temp_dir = config['temp_dir']
        self.final_dir = config['final_dir']

        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)

        self.correspondences = self._load_or_init_state()
        self.transform_matrix = self._load_matrix()

    def _load_or_init_state(self):
        """Load correspondences from disk, or return empty list if not available."""
        try:
            return load_temp_correspondences(self.temp_dir)
        except Exception:
            return []

    def save_to_matrix(self, matrix=None):
        """
        Save the transformation matrix to the final directory.
        If matrix is None, use the current self.transform_matrix.
        """
        if matrix is None:
            matrix = self.transform_matrix
        if matrix is not None:
            save_matrix(self.final_dir, matrix)

    def _load_matrix(self):
        """
        Load the transformation matrix from the final directory, if available.
        Returns:
            Loaded matrix as numpy.ndarray or None.
        """
        try:
            return load_matrix(self.final_dir)
        except Exception:
            return None

    def save_correspondences(self):
        """Save all current correspondences to the temporary directory."""
        save_temp_correspondences(self.temp_dir, self.correspondences)

    def add_or_update_correspondence(self, corr: dict):
        """
        Add a new correspondence or update an existing one (by keyframe and keypoint index).
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

    def update_last_floorplan(self, corr: dict):
        """
        Update the floorplan 2D coordinates of the last correspondence.
        """
        if self.correspondences and self.correspondences[-1] is corr:
            self.correspondences[-1]["floor2d"] = corr["floor2d"]
            self.save_correspondences()

    def get_correspondence_points(self):
        """
        Get all labeled 2D-3D point pairs where both floor2d and point3d are set.
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

    def compute_transform(self):
        """
        Compute and cache the transformation matrix from current correspondences.
        Returns:
            The computed transformation matrix.
        """
        points2D, points3D = self.get_correspondence_points()
        self.transform_matrix = compute_transform_matrix(points2D, points3D)
        return self.transform_matrix

    def get_landmarks_3d(self) -> np.ndarray:
        """
        Get all unique 3D landmarks observed by any keyframe.
        Returns:
            (N, 3) numpy array of unique points.
        """
        all_pts = []
        for img_data in self.kf_data.values():
            if "matched_3d" in img_data:
                all_pts.extend(tuple(map(tuple, img_data["matched_3d"])))
        # Remove duplicates
        unique_pts = np.array(list(set(all_pts)), dtype=np.float32) if all_pts else np.zeros((0, 3), dtype=np.float32)
        return unique_pts

    def get_cam_trajectory_3d(self) -> np.ndarray:
        """
        Get camera trajectory as array of camera centers in world coordinates.
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

    def get_current_camera_pose(self, key=None):
        """
        Get the camera pose for the given keyframe.
        Returns:
            {"center": t, "R": R} or identity/default if not found.
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

    def get_current_observed_points(self, key=None) -> np.ndarray:
        """
        Get all 3D points observed by the specified keyframe.
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
