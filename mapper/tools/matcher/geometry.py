import numpy as np
from typing import List, Dict, Set, Tuple

def _compute_fov_overlap_mask(rotations: np.ndarray, fov_deg: float = 90.0) -> np.ndarray:
    """
    Compute a boolean mask indicating which pairs of camera rotations have overlapping fields of view (FOV).

    Args:
        rotations (np.ndarray): Array of shape (N, 3, 3) representing camera rotation matrices.
        fov_deg (float): Field of view in degrees (default: 90.0).

    Returns:
        np.ndarray: Boolean (N, N) mask where True means two views' FOV overlap.
    """
    forward_vectors = rotations[:, :, 2]  # Forward direction is the third column
    forward_vectors = forward_vectors / np.linalg.norm(forward_vectors, axis=1, keepdims=True)
    cos_fov_half = np.cos(np.deg2rad(fov_deg / 2.0))
    cos_sim_matrix = forward_vectors @ forward_vectors.T
    mask = cos_sim_matrix >= cos_fov_half
    np.fill_diagonal(mask, False)
    return mask

def fast_find_adjacent_and_pose_pairs(
    img_names: List[str],
    poses: Dict[str, np.ndarray],
    gv_threshold_pos: float = 0.01,
    gv_threshold_angle_deg: float = 10.0,
    gv_fov_deg: float = 90.0,
    yaw_step_num: int = 18
) -> Set[Tuple[str, str]]:
    """
    Fast computation of candidate image pairs for local feature matching.
    Pairs are selected if they are:
      - Adjacent in yaw/pitch for the same keyframe (strict adjacency), OR
      - Within specified pose (position and angle) similarity thresholds,
      AND
      - Their camera view directions have FOV overlap.

    Args:
        img_names (List[str]): List of image filenames (should follow naming convention).
        poses (Dict[str, np.ndarray]): Mapping from image name to 4x4 pose (camera-to-world).
        gv_threshold_pos (float): Max Euclidean distance for pose similarity.
        gv_threshold_angle_deg (float): Max angular difference in degrees for pose similarity.
        gv_fov_deg (float): Field of view angle in degrees for overlap computation.
        yaw_step_num (int): Number of yaw slices (for degree calculation from index).

    Returns:
        Set[Tuple[str, str]]: Set of image name pairs (img1, img2).
    """
    def parse_pose_name(name: str) -> Tuple[int, int, int]:
        base, pitch_yaw = name.split('_pitch')
        pitch, yaw_ext = pitch_yaw.split('_yaw')
        yaw = yaw_ext.split('.')[0]
        return int(base), int(pitch), int(yaw)

    # Gather pose, yaw, pitch arrays
    poses_list = np.stack([poses[name] for name in img_names], axis=0)
    positions = poses_list[:, :3, 3]    # (N, 3)
    rotations = poses_list[:, :3, :3]   # (N, 3, 3)

    ids, pitches, yaws = zip(*[parse_pose_name(name) for name in img_names])
    ids = np.array(ids)
    pitches = np.array(pitches)
    yaws = np.array(yaws)

    YAW_STEP_DEG = 360.0 / yaw_step_num
    yaws_deg = yaws * YAW_STEP_DEG

    # --- Position and rotation similarity ---
    pos_dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)  # (N, N)
    # Compute rotation angle differences (in degrees) using trace formula
    rel_rot = np.einsum('nij,nkj->nik', rotations, rotations.transpose(0, 2, 1))
    cos_angles = (np.trace(rel_rot, axis1=-2, axis2=-1) - 1) / 2
    angles_deg = np.rad2deg(np.arccos(np.clip(cos_angles, -1.0, 1.0)))

    pose_similar_mask = (pos_dists < gv_threshold_pos) & (angles_deg < gv_threshold_angle_deg)
    np.fill_diagonal(pose_similar_mask, False)

    # --- Adjacency in yaw/pitch for the same keyframe ---
    id_mask = ids[:, None] == ids[None, :]
    pitch_diff = np.abs(pitches[:, None] - pitches[None, :]) <= 1
    yaw_diff = np.abs((yaws_deg[:, None] - yaws_deg[None, :] + 180) % 360 - 180) <= YAW_STEP_DEG
    adjacency_mask = id_mask & pitch_diff & yaw_diff
    np.fill_diagonal(adjacency_mask, False)

    # --- FOV overlap mask ---
    fov_mask = _compute_fov_overlap_mask(rotations, fov_deg=gv_fov_deg)

    # --- Final candidate mask: adjacency OR pose similarity, AND FOV overlap ---
    final_mask = (pose_similar_mask | adjacency_mask) & fov_mask

    # Upper triangular only to avoid duplicate pairs
    i_idx, j_idx = np.where(np.triu(final_mask, k=1))
    pairs = set((img_names[i], img_names[j]) for i, j in zip(i_idx, j_idx))

    print(f"[✅] Fast adjacency + pose similar + FOV overlap pairs found: {len(pairs)}")
    return pairs