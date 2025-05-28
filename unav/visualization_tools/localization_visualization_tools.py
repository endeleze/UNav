import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from matplotlib.patches import ConnectionPatch
from unav.localizer.tools.pnp import transform_pose_to_floorplan

# 可视化当前帧的局部特征点
def visualize_local_keypoints_on_image(
    img_rgb: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    figsize=(8, 8)
):
    """
    Visualize detected local keypoints on an input image, colored by their score.

    Args:
        img_rgb (np.ndarray): Input image (RGB, H x W x 3).
        keypoints (np.ndarray): Array of 2D keypoints, shape [N, 2].
        scores (np.ndarray): Keypoint scores, shape [N].
        figsize (tuple): Matplotlib figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img_rgb)
    scatter = ax.scatter(keypoints[:, 0], keypoints[:, 1], s=8, c=scores, cmap='jet', alpha=0.85)
    ax.set_title(f'Local Keypoints Visualization\nTotal: {len(keypoints)}')
    ax.axis('off')
    # 美化colorbar，仅与图片同高
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label('Keypoint Score')
    plt.tight_layout()
    plt.show()

def plot_topk_vpr_candidates(top_candidates, k=5, root_dir="/mnt/data/UNav-IO/temp"):
    """
    Visualize top-K VPR candidates side by side from /mnt/data/UNav-IO/temp/.../perspectives.
    Args:
        top_candidates: List of (map_key, img_name, score)
        k: Number of candidates to display
        root_dir: Base directory containing perspectives folders
    """
    fig, axes = plt.subplots(1, k, figsize=(4*k, 4))
    for i in range(k):
        map_key, img_name, score = top_candidates[i]
        # 解析 map_key
        try:
            place, building, floor = map_key.split("__")
        except Exception as e:
            print(f"[ERROR] map_key parsing failed for {map_key}: {e}")
            continue
        img_dir = os.path.join(root_dir, place, building, floor, "perspectives")
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        ax = axes[i]
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        else:
            ax.text(0.5, 0.5, "Not Found", ha='center', va='center', fontsize=12)
            ax.set_facecolor("gray")
        ax.set_title(f"{img_name}\nscore={score:.3f}")
        ax.axis('off')
    plt.suptitle(f"VPR Top-{k} Retrieval Results", fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_candidates_on_floorplans_with_heading(
    top_candidates,
    localizer,
    candidates_data,
    k=10,
    root_dir="/mnt/data/UNav-IO/data"
):
    """
    Visualize top-K VPR candidates' camera positions and headings on each floorplan.
    Each candidate is plotted with an auto-scaled heading arrow (like plot_camera_on_floorplan).
    """
    # 1. Group candidates by map_key
    key_to_candidates = {}
    for map_key, img_name, score in top_candidates[:k]:
        key_to_candidates.setdefault(map_key, []).append((img_name, score))

    n_keys = len(key_to_candidates)
    fig = plt.figure(figsize=(8 * n_keys, 8))

    for idx, (map_key, candidates) in enumerate(key_to_candidates.items()):
        # Load floorplan and transform matrix
        try:
            place, building, floor = map_key.split("__")
        except Exception as e:
            print(f"[WARNING] Failed to split map_key: {map_key}, {e}")
            continue
        floorplan_path = os.path.join(root_dir, place, building, floor, "floorplan.png")
        transform_matrix = localizer.transform_matrices.get(map_key)
        if not os.path.exists(floorplan_path) or transform_matrix is None:
            print(f"[WARNING] Skip {map_key} (missing floorplan or transform)")
            continue
        floorplan_img = cv2.imread(floorplan_path)
        floorplan_img = cv2.cvtColor(floorplan_img, cv2.COLOR_BGR2RGB)
        arrow_len = compute_auto_arrow_length(floorplan_img)

        ax = fig.add_subplot(1, n_keys, idx + 1)
        ax.imshow(floorplan_img)
        ax.set_title(f"Floorplan: {map_key}")

        for img_name, score in candidates:
            candidate = candidates_data.get(img_name)
            if candidate is None: continue
            ref_frame = candidate['frame']
            tvec = np.array(ref_frame['tvec'])
            qvec = np.array(ref_frame['qvec'])

            fp_pose = transform_pose_to_floorplan(qvec, tvec, transform_matrix)
            xy = fp_pose["xy"]
            ang = fp_pose["ang"]
            if xy is None or ang is None:
                continue

            # Draw camera position
            ax.plot(xy[0], xy[1], 'ro', markersize=7)
            # Draw heading arrow
            angle_rad = np.deg2rad(ang)
            dx = arrow_len * np.cos(angle_rad)
            dy = arrow_len * np.sin(angle_rad)
            ax.arrow(
                xy[0], xy[1], dx, dy,
                head_width=arrow_len * 0.2, head_length=arrow_len * 0.17,
                fc='red', ec='red', linewidth=2, alpha=0.7
            )

    plt.tight_layout()
    plt.show()
    
def visualize_query_candidate_matches(
    query_img: np.ndarray,
    query_kpts: np.ndarray,
    results: list,
    all_candidates_kpts: dict,
    root_dir: str,
    num_pairs: int = 3,
    figsize: tuple = (12, 5)
):
    """
    Visualize matches between query and candidate images, showing lines between matched keypoints.
    Supports map_key-based multi-floor datasets.

    Args:
        query_img (np.ndarray): Query image (H, W, 3).
        query_kpts (np.ndarray): Query keypoints [N, 2].
        results (list): Each dict with 'map_key', 'ref_image_name', 'query_idxs', 'ref_idxs'.
        all_candidates_kpts (dict): {map_key: {ref_image_name: keypoints [N,2]}}
        root_dir (str): UNav-IO data root, e.g. "/mnt/data/UNav-IO/temp"
        num_pairs (int): Number of pairs to visualize.
        figsize (tuple): Figure size per pair.
    """
    if not results:
        print("[WARNING] No results to visualize.")
        return
    n_show = min(num_pairs, len(results))
    indices = random.sample(range(len(results)), n_show) if len(results) > n_show else list(range(n_show))

    for idx in indices:
        res = results[idx]
        map_key = res['map_key']
        ref_name = res['ref_image_name']
        query_idxs = np.array(res['query_idxs'])
        ref_idxs = np.array(res['ref_idxs'])

        # Parse map_key to get path
        try:
            place, building, floor = map_key.split("__")
        except Exception as e:
            print(f"[ERROR] map_key parsing failed for {map_key}: {e}")
            continue
        img_dir = os.path.join(root_dir, place, building, floor, "perspectives")
        ref_img_path = os.path.join(img_dir, ref_name)
        ref_img = cv2.imread(ref_img_path)
        if ref_img is None:
            print(f"[WARNING] Candidate image not found: {ref_img_path}")
            continue
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        # Candidate keypoints from all_candidates_kpts
        if map_key not in all_candidates_kpts or ref_name not in all_candidates_kpts[map_key]:
            print(f"[WARNING] Keypoints not found for: {map_key} - {ref_name}")
            continue
        cand_kpts = all_candidates_kpts[map_key][ref_name]

        # Prepare canvas
        h1, w1 = query_img.shape[:2]
        h2, w2 = ref_img.shape[:2]
        out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        out_img[:h1, :w1] = query_img
        out_img[:h2, w1:w1 + w2] = ref_img

        plt.figure(figsize=figsize)
        plt.imshow(out_img)
        plt.axis('off')

        # Get matched keypoints (in original images)
        q_pts = query_kpts[query_idxs]
        r_pts = cand_kpts[ref_idxs]
        r_pts_offset = r_pts + np.array([w1, 0])

        # Draw lines
        for pt1, pt2 in zip(q_pts, r_pts_offset):
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c='orange', linewidth=1.2, alpha=0.8)

        # Draw keypoints
        plt.scatter(q_pts[:, 0], q_pts[:, 1], c='cyan', s=18, label='Query')
        plt.scatter(r_pts_offset[:, 0], r_pts_offset[:, 1], c='lime', s=18, label='Candidate')

        title = f"Query ↔ {ref_name} ({map_key})\nMatches: {len(q_pts)}"
        plt.title(title)
        plt.show()
        
def visualize_2d_3d_crosslink(
    query_img,
    image_points,
    object_points,
    transform_matrix,
    floorplan_img,
    num_matches=10,
    crop_size=300,
    random_seed=42
):
    """
    Visualize 2D-3D matches with lines connecting query image and floorplan projections.

    Args:
        query_img (np.ndarray): Query image (H,W,3).
        image_points (np.ndarray): [N,2] matched 2D points in image.
        object_points (np.ndarray): [N,3] matched 3D points in map.
        transform_matrix (np.ndarray): [2,4] for 3D->2D projection.
        floorplan_img (np.ndarray): Floorplan image.
        num_matches (int): Number of matches to visualize.
        crop_size (int): Cropping region size for floorplan.
        random_seed (int): For reproducibility.
    """
    assert image_points.shape[0] == object_points.shape[0]
    N = image_points.shape[0]
    idx = np.arange(N)
    random.seed(random_seed)
    if N > num_matches:
        idx = random.sample(list(idx), num_matches)
    # Project 3D→2D
    obj3d_h = np.concatenate([object_points[idx], np.ones((len(idx),1))], axis=1)
    proj_2d = (transform_matrix @ obj3d_h.T).T

    # Crop floorplan
    cx, cy = np.mean(proj_2d, axis=0)
    h, w = floorplan_img.shape[:2]
    x1 = int(max(cx - crop_size//2, 0))
    y1 = int(max(cy - crop_size//2, 0))
    x2 = int(min(cx + crop_size//2, w))
    y2 = int(min(cy + crop_size//2, h))
    floorplan_crop = floorplan_img[y1:y2, x1:x2]
    proj_2d_crop = proj_2d - np.array([x1, y1])
    # Resize query img to same width
    qh, qw = query_img.shape[:2]
    scale = (x2-x1) / qw
    img_resized = cv2.resize(query_img, (x2-x1, int(qh*scale)))
    img_points_scaled = image_points[idx] * scale
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    # Query image
    ax1.imshow(img_resized)
    ax1.scatter(img_points_scaled[:,0], img_points_scaled[:,1], c='r', s=40, label='2D Points')
    ax1.set_title('Query Image')
    ax1.axis('off')
    # Floorplan crop
    if floorplan_crop.ndim == 2:
        ax2.imshow(floorplan_crop, cmap='gray')
    else:
        ax2.imshow(floorplan_crop)
    ax2.scatter(proj_2d_crop[:,0], proj_2d_crop[:,1], c='b', s=40, label='Projected 3D→2D')
    ax2.set_title('Projected 3D Points on Floorplan (Local Crop)')
    ax2.axis('off')
    # Draw cross-connections
    for i in range(len(idx)):
        con = ConnectionPatch(
            xyA=(proj_2d_crop[i,0], proj_2d_crop[i,1]), coordsA=ax2.transData,
            xyB=(img_points_scaled[i,0], img_points_scaled[i,1]), coordsB=ax1.transData,
            color="green", lw=1.6, alpha=0.8
        )
        fig.add_artist(con)
    plt.tight_layout()
    plt.show()
    
def plot_camera_on_floorplan(
    floorplan_img: np.ndarray,
    cam_xy: tuple,
    cam_angle: float,
    marker: str = 'o',
    show: bool = True,
    ax: plt.Axes = None,
    figsize: tuple = (10, 8),
    color: str = 'red'
):
    """
    Draw the camera location and orientation on the floorplan.

    Args:
        floorplan_img (np.ndarray): Floorplan image (H, W, 3 or 1).
        cam_xy (tuple): (x, y) in floorplan pixel coordinates.
        cam_angle (float): Angle (in degrees), 0 means +x axis.
        marker (str): Matplotlib marker for camera location.
        show (bool): If True, calls plt.show().
        ax (plt.Axes): If provided, draws on this axis.
        figsize (tuple): Figure size if ax is None.
        color (str): Arrow and marker color.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        own_fig = True
    else:
        own_fig = False

    arrow_len = compute_auto_arrow_length(floorplan_img)

    # Show floorplan (grayscale or color)
    if floorplan_img.ndim == 2:
        ax.imshow(floorplan_img, cmap='gray')
    else:
        ax.imshow(floorplan_img)
    # Draw camera location
    ax.plot(cam_xy[0], cam_xy[1], marker, color=color, markersize=14)
    # Draw orientation arrow
    angle_rad = np.deg2rad(cam_angle)
    dx = arrow_len * np.cos(angle_rad)
    dy = arrow_len * np.sin(angle_rad)
    ax.arrow(
        cam_xy[0], cam_xy[1], dx, dy,
        head_width=arrow_len*0.3, head_length=arrow_len*0.25,
        fc=color, ec=color, linewidth=5
    )
    ax.set_title("Estimated Camera Location on Floorplan")
    ax.axis('off')
    if own_fig and show:
        plt.show()
    return ax

def compute_auto_arrow_length(floorplan_img, ratio=0.05):
    """Auto-compute an arrow length based on floorplan size."""
    h, _ = floorplan_img.shape[:2]
    length = int(ratio * h)
    return length