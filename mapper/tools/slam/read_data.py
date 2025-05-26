import os
import re
import sqlite3
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_keyframe_image_list(keyframe_dir: str) -> List[str]:
    """
    Returns sorted list of files matching image\\d+\\.png in keyframe_dir.
    """
    files = os.listdir(keyframe_dir)
    imgs = [f for f in files if re.match(r'image\d+\.png$', f)]
    return sorted(imgs, key=lambda x: int(re.findall(r'\d+', x)[0]))

def read_keyframe_trajectory(
    trajectory_file: str,
    keyframe_names: List[str]
) -> Dict[str, np.ndarray]:
    """
    Reads trajectory file lines of format:
        frame_id tx ty tz qx qy qz qw
    (camera→world quaternion), and returns dict of 4×4 camera→world poses.
    """
    poses: Dict[str, np.ndarray] = {}
    with open(trajectory_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < len(keyframe_names):
        raise ValueError(
            f"[Error] {len(lines)} poses vs {len(keyframe_names)} keyframes"
        )
    for idx, line in enumerate(lines[:len(keyframe_names)]):
        tok = line.split()
        if len(tok) != 8:
            print(f"[Warning] skipping invalid trajectory line: {line}")
            continue
        _, tx, ty, tz, qx, qy, qz, qw = tok
        t = np.array([float(tx), float(ty), float(tz)])
        quat = np.array([float(qx), float(qy), float(qz), float(qw)])  # x,y,z,w
        R_mat = R.from_quat(quat).as_matrix()  # camera→world
        pose = np.eye(4)
        pose[:3, :3] = R_mat
        pose[:3, 3] = t
        poses[keyframe_names[idx]] = pose
    return poses

def extract_kf_pose_and_matches(
    db_path: str, keyframe_dir: str
) -> Dict[str, Dict[str, Any]]:
    """
    Extract keyframe poses, undistorted 2D keypoints, and associated 3D landmarks
    from a stella_vslam_dense SQLite .msg map file, and map them using image file names.

    Args:
        db_path (str): Path to the SQLite .msg map file.
        keyframe_dir (str): Path to the directory containing keyframe image files (e.g., image0.png, image1.png, ...).

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping image name to:
            - 'kf_id' (int): Keyframe ID
            - 'T_cw' (np.ndarray): 4x4 camera-to-world transformation matrix
            - 'keypoints' (np.ndarray): (N, 2) 2D keypoints
            - 'matched_3d' (List[Optional[np.ndarray]]): 3D matches
    """
    # Step 0: Get list of keyframe image filenames indexed by keyframe ID
    img_name_list = get_keyframe_image_list(keyframe_dir)
    id_to_imgname = {i: name for i, name in enumerate(img_name_list)}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Step 1: Load all landmarks
    cursor.execute("SELECT id, pos_w FROM landmarks")
    landmarks = {
        lm_id: np.frombuffer(pos_blob, dtype=np.float64)
        for lm_id, pos_blob in cursor.fetchall()
    }

    # Step 2: Load associations (keypoint idx → lm_id)
    cursor.execute("SELECT id, lm_ids FROM associations")
    assoc_map = {}
    for kf_id, lm_blob in cursor.fetchall():
        lm_ids = np.frombuffer(lm_blob, dtype=np.int32)
        for i, lm_id in enumerate(lm_ids):
            if lm_id >= 0:
                assoc_map[(kf_id, i)] = lm_id

    # Step 3: Load keyframe pose and keypoints
    cursor.execute("SELECT id, pose_cw, undist_keypts, n_keypts FROM keyframes")
    kf_data = {}
    for kf_id, pose_blob, keypt_blob, n_keypts in cursor.fetchall():
        try:
            T_cw = np.frombuffer(pose_blob, dtype=np.float64).reshape(4, 4)
        except ValueError:
            T_cw = np.frombuffer(pose_blob, dtype=np.float32).reshape(4, 4)
        
        try:
            raw = np.frombuffer(keypt_blob, dtype=np.float32)
            keypoints = raw.reshape(n_keypts, 7)[:, :2]
        except Exception:
            raise ValueError(
                f"❌ Failed to parse keypoints in kf {kf_id}: expected float32 × 7, got {len(keypt_blob)} bytes."
            )

        matched_3d = [
            landmarks.get(assoc_map.get((kf_id, i))) for i in range(n_keypts)
        ]

        # Match kf_id with image name
        img_name = id_to_imgname.get(kf_id)
        if img_name is not None:
            kf_data[img_name] = {
                "kf_id": kf_id,
                "T_cw": T_cw,
                "keypoints": keypoints,
                "matched_3d": matched_3d,
            }

    conn.close()
    print(f"✅ Successfully extracted {len(kf_data)} keyframes with image names.")
    return kf_data

def filter_kf_data(
    kf_data: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Filter raw keyframe data to only keep keypoints that have a matched 3D landmark.

    Args:
        kf_data: 
            Original dict mapping image file name (e.g. "000123_05.png") to entries:
            {
                "kf_id": int,
                "T_cw": np.ndarray (4x4),
                "keypoints": np.ndarray of shape (N,2),
                "matched_3d": List[Optional[List[float]]]  # length N
            }

    Returns:
        filtered: Dict mapping the same image names to a simplified entry:
        {
            "kf_id": int,
            "T_cw": np.ndarray (4x4),
            "gp": List[Tuple[float, float]],    # only 2D points with a match
            "lm": List[List[float]]             # corresponding 3D landmarks
        }
        Only images with at least one valid match are included.
    """
    filtered: Dict[str, Dict[str, Any]] = {}
    for name, entry in kf_data.items():
        keypoints = entry.get("keypoints", [])
        matched_3d = entry.get("matched_3d", [])
        gp: List[Tuple[float, float]] = []
        lm: List[List[float]] = []

        # collect only those keypoints with a non-None 3D match
        for pt, m in zip(keypoints, matched_3d):
            if m is not None:
                x, y = float(pt[0]), float(pt[1])
                gp.append((x, y))
                lm.append(m)

        if gp:
            filtered[name] = {
                "kf_id":   entry["kf_id"],
                "T_cw":    entry["T_cw"],
                "keypoints":      gp,
                "matched_3d":      lm,
            }

    return filtered