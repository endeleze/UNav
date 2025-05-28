import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
import open3d as o3d

from unav.core.colmap.read_write_model import read_model, qvec2rotmat

def visualize_point_cloud_views(
    pcd_path: str,
    std_threshold: float = 3.0,
    show_filtered_ratio: bool = True
) -> None:
    """
    Visualize a point cloud in three orthogonal projections: XY (top), XZ (front), and YZ (side).
    Optionally removes outlier points based on distance from the centroid using a standard deviation threshold.

    Args:
        pcd_path (str): Path to the point cloud file (supported by Open3D, e.g., .ply, .pcd).
        std_threshold (float): Points further than `std_threshold` × std from the centroid are removed.
        show_filtered_ratio (bool): If True, print ratio of points retained after filtering.
    """
    def remove_outliers_by_std(points: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes points whose distance to the centroid exceeds threshold × standard deviation.

        Args:
            points (np.ndarray): Array of shape (N, 3) representing point coordinates.
            threshold (float): Standard deviation multiplier for outlier removal.

        Returns:
            filtered_points (np.ndarray): Filtered points of shape (M, 3).
            keep_mask (np.ndarray): Boolean mask indicating points to keep.
        """
        center = np.mean(points, axis=0)
        dists = np.linalg.norm(points - center, axis=1)
        dist_std = np.std(dists)
        keep_mask = dists < threshold * dist_std
        return points[keep_mask], keep_mask

    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Remove outliers
    filtered_points, keep_mask = remove_outliers_by_std(points, std_threshold)
    filtered_colors = colors[keep_mask]

    if show_filtered_ratio:
        ratio = 100 * len(filtered_points) / len(points)
        print(f"[INFO] Points before filtering: {len(points)}, after: {len(filtered_points)} ({ratio:.1f}%)")

    # Visualize projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].scatter(filtered_points[:, 0], filtered_points[:, 1], c=filtered_colors, s=0.1)
    axes[0].set_title(f"Top View (XY)\n(std_thresh={std_threshold})")
    axes[0].axis('off')

    axes[1].scatter(filtered_points[:, 0], filtered_points[:, 2], c=filtered_colors, s=0.1)
    axes[1].set_title(f"Front View (XZ)\n(std_thresh={std_threshold})")
    axes[1].axis('off')

    axes[2].scatter(filtered_points[:, 1], filtered_points[:, 2], c=filtered_colors, s=0.1)
    axes[2].set_title(f"Side View (YZ)\n(std_thresh={std_threshold})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def plot_camera_trajectory_xz(
    trajectory_file: str,
    title: str = "Camera Trajectory (XZ Plane, Normalized)",
    figsize: Tuple[int, int] = (8, 8),
    show_axis: bool = True,
    show: bool = True,
    save_path: Optional[str] = None,
    target_scale: float = 10.0
) -> None:
    """
    Plot a normalized camera trajectory in the XZ plane.

    Args:
        trajectory_file (str): Path to trajectory text file (TUM or COLMAP format).
        title (str): Plot title.
        figsize (Tuple[int, int]): Figure size for matplotlib.
        show_axis (bool): If True, plot local camera axes for a subset of poses.
        show (bool): Whether to display the plot.
        save_path (Optional[str]): Optional path to save the plot.
        target_scale (float): Scale for normalization (largest extent will match this value).
    """
    poses = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = tokens
            t = np.array([float(tx), float(ty), float(tz)])
            q = np.array([float(qx), float(qy), float(qz), float(qw)])
            R_mat = R.from_quat(q).as_matrix()
            pose = np.eye(4)
            pose[:3, :3] = R_mat
            pose[:3, 3] = t
            poses.append(pose)

    if not poses:
        print(f"[ERROR] No valid poses found in {trajectory_file}")
        return

    cams_xyz = np.array([p[:3, 3] for p in poses])
    cams_xz = cams_xyz[:, [0, 2]]
    center = np.mean(cams_xz, axis=0)
    cams_xz_centered = cams_xz - center

    max_extent = np.max(np.linalg.norm(cams_xz_centered, axis=1))
    scale_factor = target_scale / max_extent if max_extent > 0 else 1.0
    cams_xz_scaled = cams_xz_centered * scale_factor

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cams_xz_scaled[:, 0], cams_xz_scaled[:, 1], 'o-', label='Camera Centers (Normalized XZ)', markersize=3)

    if show_axis:
        for p in poses[::max(1, len(poses)//50)]:
            center_orig = p[:3, 3]
            center_scaled = (center_orig[[0, 2]] - center) * scale_factor
            axes = p[:3, :3] * 0.01 * scale_factor
            # X axis (green)
            x_dir = axes[[0, 2], 0]
            ax.arrow(center_scaled[0], center_scaled[1], x_dir[0], x_dir[1], head_width=0.2, color='green', alpha=0.6)
            # Z axis (red)
            z_dir = axes[[0, 2], 2]
            ax.arrow(center_scaled[0], center_scaled[1], z_dir[0], z_dir[1], head_width=0.2, color='r', alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Z (normalized)')
    ax.legend()
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Saved normalized trajectory XZ plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_colmap_poses_on_xz(
    images_txt_path: str,
    scale: float = 0.3,
    target_scale: float = 1.0,
    figsize: Tuple[int, int] = (8, 8),
    show_axis: bool = True,
    show: bool = True,
    save_path: Optional[str] = None,
    title: str = "Camera Poses (yaw00 / y0, Normalized XZ)"
) -> None:
    """
    Plot COLMAP camera poses with suffix '_yaw00' or '_y0' on the normalized XZ plane.

    Args:
        images_txt_path (str): Path to COLMAP images.txt file.
        scale (float): Arrow scaling for axis visualization.
        target_scale (float): Scale for normalization.
        figsize (Tuple[int, int]): Figure size.
        show_axis (bool): If True, draw local X/Z axes.
        show (bool): Whether to display the plot.
        save_path (Optional[str]): Optional path to save the plot.
        title (str): Plot title.
    """
    def is_yaw00_or_y0(name: str) -> bool:
        return re.search(r'(?:_y0(?:\.png)?|_yaw0?0(?:\.png)?)$', name) is not None

    cam_positions = []
    cam_rotations = []

    with open(images_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 10:
                continue
            img_name = tokens[-1]
            if not is_yaw00_or_y0(img_name):
                continue
            qw, qx, qy, qz = map(float, tokens[1:5])
            tx, ty, tz = map(float, tokens[5:8])
            R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t_wc = np.array([tx, ty, tz])
            # Camera center in world coordinates
            C = -R_wc.T @ t_wc
            R_cw = R_wc.T
            cam_positions.append(C)
            cam_rotations.append(R_cw)

    if not cam_positions:
        print(f"[ERROR] No valid matching images found in {images_txt_path}")
        return

    cam_positions = np.array(cam_positions)
    cam_rotations = np.array(cam_rotations)
    cams_xz = cam_positions[:, [0, 2]]
    center = np.mean(cams_xz, axis=0)
    cams_xz_centered = cams_xz - center
    max_extent = np.max(np.linalg.norm(cams_xz_centered, axis=1))
    scale_factor = target_scale / max_extent if max_extent > 0 else 1.0
    cams_xz_scaled = cams_xz_centered * scale_factor

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cams_xz_scaled[:, 0], cams_xz_scaled[:, 1], 'o-', label='Camera Centers (Normalized XZ)', markersize=3)

    if show_axis:
        step = max(1, len(cam_positions) // 50)
        for i in range(0, len(cam_positions), step):
            R_cw = cam_rotations[i]
            center_orig = cam_positions[i]
            center_scaled = (center_orig[[0, 2]] - center) * scale_factor
            axes = R_cw * 0.1 * scale_factor
            # X axis (green)
            x_dir = axes[[0, 2], 0]
            ax.arrow(center_scaled[0], center_scaled[1], x_dir[0], x_dir[1], head_width=0.02, color='green', alpha=0.6)
            # Z axis (red)
            z_dir = axes[[0, 2], 2]
            ax.arrow(center_scaled[0], center_scaled[1], z_dir[0], z_dir[1], head_width=0.02, color='red', alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Z (normalized)')
    ax.legend()
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_colmap_images_trajectory_xz(
    image_file: str,
    figsize: Tuple[int, int] = (8, 8),
    show_axis: bool = False,
    axis_limit: Optional[float] = None,
    title: str = "COLMAP Images Trajectory (XZ Plane)"
) -> None:
    """
    Plot camera trajectory in the XZ plane from a COLMAP images.txt file.

    Args:
        image_file (str): Path to COLMAP images.txt.
        figsize (Tuple[int, int]): Matplotlib figure size.
        show_axis (bool): If True, draw camera local axes.
        axis_limit (Optional[float]): Optional axis limit for plot range.
        title (str): Plot title.
    """
    centers, rotations = [], []
    with open(image_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            if len(tokens) < 10:
                continue
            tx, ty, tz = map(float, tokens[5:8])
            qw, qx, qy, qz = map(float, tokens[1:5])
            centers.append([tx, ty, tz])
            R_cw = R.from_quat([qx, qy, qz, qw]).as_matrix()
            rotations.append(R_cw)

    centers = np.array(centers)
    if centers.size == 0:
        print(f"[ERROR] No valid entries found in {image_file}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(centers[:, 0], centers[:, 2], 'o-', label='Camera Centers (XZ)', markersize=4)

    if show_axis:
        step = max(1, len(centers) // 50)
        for i in range(0, len(centers), step):
            center = centers[i, [0, 2]]
            R_cw = rotations[i]
            x_dir = R_cw[:, 0][[0, 2]] * 0.2
            ax.arrow(center[0], center[1], x_dir[0], x_dir[1], head_width=0.05, color='r', alpha=0.6)
            z_dir = R_cw[:, 2][[0, 2]] * 0.2
            ax.arrow(center[0], center[1], z_dir[0], z_dir[1], head_width=0.05, color='b', alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    if axis_limit is not None:
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
    plt.tight_layout()
    plt.show()


def show_sample_images_with_pose(
    image_dir: str,
    image_file: str,
    num_images: int = 3
) -> None:
    """
    Display random sample images with their pose information parsed from COLMAP images.txt.

    Args:
        image_dir (str): Directory containing images.
        image_file (str): Path to COLMAP images.txt.
        num_images (int): Number of samples to display.
    """
    if not os.path.exists(image_file):
        print(f"[ERROR] Images file not found: {image_file}")
        return

    image_pose_map = {}
    with open(image_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens or tokens[0].startswith('#') or len(tokens) < 10:
                continue
            image_name = tokens[9]
            pose_values = [float(val) for val in tokens[1:8]]
            qw, qx, qy, qz, tx, ty, tz = pose_values
            pose_str = (
                f"tx: {tx:.2f} ty: {ty:.2f} tz: {tz:.2f}\n"
                f"qw: {qw:.2f} qx: {qx:.2f} qy: {qy:.2f} qz: {qz:.2f}"
            )
            image_pose_map[image_name] = pose_str

    output_images = sorted([f for f in os.listdir(image_dir) if f in image_pose_map])
    print(f"[INFO] Total images found with pose: {len(output_images)}")
    sample_files = output_images[:num_images]

    fig, axes = plt.subplots(1, len(sample_files), figsize=(5 * len(sample_files), 4))
    if len(sample_files) == 1:
        axes = [axes]
    for ax, file in zip(axes, sample_files):
        img = cv2.imread(os.path.join(image_dir, file))
        if img is None:
            print(f"[WARNING] Failed to read {file}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"{file}\n{image_pose_map[file]}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_local_keypoints(
    img_name: str,
    image_dir: str,
    local_feature_file: str,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Visualize detected local keypoints on an image, colored by score.

    Args:
        img_name (str): Image file name (must be present in both directory and H5 file).
        image_dir (str): Directory containing images.
        local_feature_file (str): Path to H5 file storing local features.
        figsize (Tuple[int, int]): Matplotlib figure size.
    """
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Failed to read image: {img_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with h5py.File(local_feature_file, 'r') as h5_file:
        if img_name not in h5_file:
            print(f"[ERROR] Image key not found in local feature file: {img_name}")
            return
        kp = h5_file[img_name]['keypoints'][:]
        scores = h5_file[img_name]['scores'][:]

    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.scatter(kp[:, 0], kp[:, 1], s=5, c=scores, cmap='jet', alpha=0.75)
    plt.colorbar(label='Keypoint Score')
    plt.title(f'Keypoints Visualization for {img_name}\nTotal Keypoints: {kp.shape[0]}')
    plt.axis('off')
    plt.show()


def read_colmap_local_feature(local_feature_file: str) -> Dict[str, np.ndarray]:
    """
    Read local features from COLMAP-style text file.

    Args:
        local_feature_file (str): Path to feature file.

    Returns:
        Dict[str, np.ndarray]: Mapping from image name to keypoints (N,2).
    """
    features = {}
    with open(local_feature_file, 'r') as f:
        lines = f.read().split('\n')
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line == '':
            idx += 1
            continue
        parts = line.strip().split()
        image_name, n_kpts = parts[0], int(parts[1])
        kpts = []
        for _ in range(n_kpts):
            idx += 1
            kpt_line = lines[idx].strip()
            x, y = map(float, kpt_line.split())
            kpts.append([x, y])
        features[image_name] = np.array(kpts)
        idx += 1
    return features


def read_colmap_matches(matches_file: str) -> List[Tuple[str, str, np.ndarray]]:
    """
    Read matches from COLMAP-style text file.

    Args:
        matches_file (str): Path to matches file.

    Returns:
        List[Tuple[str, str, np.ndarray]]: Each tuple is (img1, img2, matches_array[N,2]).
    """
    matches = []
    with open(matches_file, 'r') as f:
        lines = f.read().split('\n')
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line == '':
            idx += 1
            continue
        parts = line.split()
        img1, img2 = parts[0], parts[1]
        pair_matches = []
        while True:
            idx += 1
            if idx >= len(lines):
                break
            match_line = lines[idx].strip()
            if match_line == '':
                break
            u, v = map(int, match_line.split())
            pair_matches.append([u, v])
        matches.append((img1, img2, np.array(pair_matches)))
    return matches


def visualize_random_matches(
    local_feature_file: Union[str, Path],
    matches_file: Union[str, Path],
    image_dir: Union[str, Path],
    num_pairs: int = 3,
    colormap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Visualize random feature matches from COLMAP-style H5 files.

    Args:
        local_feature_file (Union[str, Path]): Path to local features H5 file.
        matches_file (Union[str, Path]): Path to matches H5 file.
        image_dir (Union[str, Path]): Directory containing images.
        num_pairs (int): Number of random image pairs to visualize.
        colormap (str): Matplotlib colormap for match scores.
        figsize (Tuple[int, int]): Figure size.
    """
    local_feature_file = Path(local_feature_file)
    matches_file = Path(matches_file)
    image_dir = Path(image_dir)

    assert local_feature_file.exists(), f"Local feature file not found: {local_feature_file}"
    assert matches_file.exists(), f"Matches file not found: {matches_file}"
    assert image_dir.exists(), f"Image directory not found: {image_dir}"

    with h5py.File(local_feature_file, 'r') as f_feat, h5py.File(matches_file, 'r') as f_match:
        match_keys = list(f_match.keys())
        if not match_keys:
            print("[WARNING] No matches found in the file.")
            return
        pattern = re.compile(r'(.+?\.png)_(.+?\.png)')
        selected_matches = random.sample(match_keys, min(num_pairs, len(match_keys)))

        for pair_key in selected_matches:
            match = pattern.match(pair_key)
            if not match:
                print(f"[WARNING] Invalid match key format: {pair_key}")
                continue

            img1_name, img2_name = match.group(1), match.group(2)
            img1_path = image_dir / img1_name
            img2_path = image_dir / img2_name

            if not img1_path.exists() or not img2_path.exists():
                print(f"[WARNING] Image file not found: {img1_path} or {img2_path}")
                continue

            img1 = cv2.cvtColor(cv2.imread(str(img1_path)), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(str(img2_path)), cv2.COLOR_BGR2RGB)
            kpts1 = f_feat[img1_name]['keypoints'][:]
            kpts2 = f_feat[img2_name]['keypoints'][:]
            matches = f_match[pair_key]['matches0'][:]
            scores = f_match[pair_key]['matching_scores0'][:]

            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            new_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            new_img[:h1, :w1] = img1
            new_img[:h2, w1:w1 + w2] = img2

            norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            color_map = cm.get_cmap(colormap)(norm_scores)

            plt.figure(figsize=figsize)
            plt.imshow(new_img)
            plt.axis('off')

            for (idx0, idx1), color in zip(matches, color_map):
                pt1 = kpts1[idx0]
                pt2 = kpts2[idx1] + np.array([w1, 0])
                plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=1.5)
                plt.scatter([pt1[0]], [pt1[1]], c=[color], s=20)
                plt.scatter([pt2[0]], [pt2[1]], c=[color], s=20)

            plt.title(f'{img1_name} ↔ {img2_name} with {len(matches)} matches')
            plt.show()


def render_colmap_point_cloud_views(
    model_dir: str,
    ext: str = ".bin",
    min_track_len: int = 3
) -> None:
    """
    Render COLMAP sparse point cloud and camera centers in three projections after outlier removal.

    Args:
        model_dir (str): Path to COLMAP model directory.
        ext (str): File extension for model files ('.bin' or '.txt').
        min_track_len (int): Minimum observation count to keep a 3D point.
    """
    def remove_outliers_by_std(points: np.ndarray, std_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        center = np.mean(points, axis=0)
        dists = np.linalg.norm(points - center, axis=1)
        dist_std = np.std(dists)
        keep = dists < std_threshold * dist_std
        return points[keep], keep

    cameras, images, points3D = read_model(model_dir, ext)
    print(f"[INFO] Total images: {len(images)}")
    print(f"[INFO] Total 3D points before filtering: {len(points3D)}")

    xyz = []
    rgb = []
    for p in points3D.values():
        if len(p.point2D_idxs) >= min_track_len:
            xyz.append(p.xyz)
            rgb.append(p.rgb / 255.0)
    xyz = np.array(xyz)
    rgb = np.array(rgb)

    if len(xyz) == 0:
        raise ValueError("No 3D points found")

    xyz_clean, keep_mask = remove_outliers_by_std(xyz, std_threshold=3.0)
    rgb_clean = np.array(rgb)[keep_mask]
    print(f"[INFO] 3D points after outlier removal: {len(xyz_clean)}")

    cam_centers = []
    for img in images.values():
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        C = -R.T @ t
        cam_centers.append(C)
    cam_centers = np.array(cam_centers)

    all_pts = np.vstack([xyz_clean, cam_centers])
    x_min, y_min, z_min = np.min(all_pts, axis=0)
    x_max, y_max, z_max = np.max(all_pts, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].scatter(xyz_clean[:, 0], xyz_clean[:, 1], s=0.3, c=rgb_clean)
    axes[0].scatter(cam_centers[:, 0], cam_centers[:, 1], c='red', s=2)
    axes[0].set_title("Top View (XY)")
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].axis("equal")

    axes[1].scatter(xyz_clean[:, 0], xyz_clean[:, 2], s=0.3, c=rgb_clean)
    axes[1].scatter(cam_centers[:, 0], cam_centers[:, 2], c='red', s=2)
    axes[1].set_title("Front View (XZ)")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(z_min, z_max)
    axes[1].axis("equal")

    axes[2].scatter(xyz_clean[:, 1], xyz_clean[:, 2], s=0.3, c=rgb_clean)
    axes[2].scatter(cam_centers[:, 1], cam_centers[:, 2], c='red', s=2)
    axes[2].set_title("Side View (YZ)")
    axes[2].set_xlim(y_min, y_max)
    axes[2].set_ylim(z_min, z_max)
    axes[2].axis("equal")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
