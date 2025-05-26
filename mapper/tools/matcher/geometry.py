import numpy as np

def _compute_fov_overlap_mask(rotations: np.ndarray, fov_deg: float = 90.0) -> np.ndarray:
    forward_vectors = rotations[:, :, 2]
    forward_vectors = forward_vectors / np.linalg.norm(forward_vectors, axis=1, keepdims=True)
    cos_fov_half = np.cos(np.deg2rad(fov_deg / 2.0))
    cos_sim_matrix = forward_vectors @ forward_vectors.T
    mask = cos_sim_matrix >= cos_fov_half
    np.fill_diagonal(mask, False)
    return mask

def fast_find_adjacent_and_pose_pairs(
    img_names,
    poses: dict,
    gv_threshold_pos: float = 0.01,
    gv_threshold_angle_deg: float = 10.0,
    gv_fov_deg: float = 90.0,
    yaw_step_num: int = 18
) -> set:
    def parse_pose_name(name):
        base, pitch_yaw = name.split('_pitch')
        pitch, yaw_ext = pitch_yaw.split('_yaw')
        yaw = yaw_ext.split('.')[0]
        return int(base), int(pitch), int(yaw)

    poses_list = np.stack([poses[name] for name in img_names], axis=0)
    positions = poses_list[:, :3, 3]
    rotations = poses_list[:, :3, :3]

    ids, pitches, yaws = zip(*[parse_pose_name(name) for name in img_names])
    ids = np.array(ids)
    pitches = np.array(pitches)
    yaws = np.array(yaws)

    YAW_STEP_DEG = 360.0 / yaw_step_num
    yaws_deg = yaws * YAW_STEP_DEG
    N = len(img_names)

    # Compute distance and rotation similarity
    pos_dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    angles_deg = np.rad2deg(np.arccos(np.clip((np.einsum('nij,nkj->nik', rotations, rotations.transpose(0, 2, 1)).trace(axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)))

    pose_similar_mask = (pos_dists < gv_threshold_pos) & (angles_deg < gv_threshold_angle_deg)
    np.fill_diagonal(pose_similar_mask, False)

    # Adjacency mask (strict yaw neighbor only)
    id_mask = ids[:, None] == ids[None, :]
    pitch_diff = np.abs(pitches[:, None] - pitches[None, :]) <= 1
    yaw_diff = np.abs((yaws_deg[:, None] - yaws_deg[None, :] + 180) % 360 - 180) <= YAW_STEP_DEG
    adjacency_mask = id_mask & pitch_diff & yaw_diff
    np.fill_diagonal(adjacency_mask, False)

    # FOV overlap mask (use camera orientation only)
    fov_mask = _compute_fov_overlap_mask(rotations, fov_deg=gv_fov_deg)

    # Combine strictly
    final_mask = (pose_similar_mask | adjacency_mask) & fov_mask

    i_idx, j_idx = np.where(np.triu(final_mask, k=1))
    pairs = set((img_names[i], img_names[j]) for i, j in zip(i_idx, j_idx))

    print(f"[✅] Fast adjacency + pose similar + FOV overlap pairs found: {len(pairs)}")
    return pairs