import os
import re
import sqlite3
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_keyframe_image_list(keyframe_dir: str) -> List[str]:
    """
    Returns a naturally sorted list of image files matching 'image\\d+\\.png' in the specified directory.
    """
    files = os.listdir(keyframe_dir)
    imgs = [f for f in files if re.match(r'image\d+\.png$', f)]
    return sorted(imgs, key=lambda x: int(re.findall(r'\d+', x)[0]))

def read_keyframe_trajectory(
    trajectory_file: str,
    keyframe_names: List[str]
) -> Dict[str, np.ndarray]:
    """
    Reads camera trajectory from a text file, mapping keyframe names to camera→world pose matrices.

    Args:
        trajectory_file (str): Path to trajectory file, lines formatted as:
            frame_id tx ty tz qx qy qz qw
        keyframe_names (List[str]): Ordered list of keyframe image names.

    Returns:
        Dict[str, np.ndarray]: image name → 4x4 camera-to-world pose matrix.
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
            print(f"[Warning] Skipping invalid trajectory line: {line}")
            continue
        _, tx, ty, tz, qx, qy, qz, qw = tok
        t = np.array([float(tx), float(ty), float(tz)])
        quat = np.array([float(qx), float(qy), float(qz), float(qw)])  # [x, y, z, w]
        R_mat = R.from_quat(quat).as_matrix()  # scipy expects [x, y, z, w]
        pose = np.eye(4)
        pose[:3, :3] = R_mat
        pose[:3, 3] = t
        poses[keyframe_names[idx]] = pose
    return poses

def extract_kf_pose_and_matches(
    db_path: str, keyframe_dir: str
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts keyframe poses, 2D keypoints, and associated 3D landmark matches from a stella_vslam_dense SQLite map.

    Args:
        db_path (str): Path to the .msg SQLite map.
        keyframe_dir (str): Directory with keyframe images.

    Returns:
        Dict[str, Dict]: image name → {
            'kf_id': int,
            'T_cw': 4x4 np.ndarray,
            'keypoints': np.ndarray (N,2),
            'matched_3d': List[Optional[np.ndarray]],
        }
    """
    # 1. Index keyframe images by their expected frame id (sequential)
    img_name_list = get_keyframe_image_list(keyframe_dir)
    id_to_imgname = {i: name for i, name in enumerate(img_name_list)}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 2. Load all landmarks (id → position)
    cursor.execute("SELECT id, pos_w FROM landmarks")
    landmarks = {
        lm_id: np.frombuffer(pos_blob, dtype=np.float64)
        for lm_id, pos_blob in cursor.fetchall()
    }

    # 3. Load associations: keypoint index → landmark id
    cursor.execute("SELECT id, lm_ids FROM associations")
    assoc_map = {}
    for kf_id, lm_blob in cursor.fetchall():
        lm_ids = np.frombuffer(lm_blob, dtype=np.int32)
        for i, lm_id in enumerate(lm_ids):
            if lm_id >= 0:
                assoc_map[(kf_id, i)] = lm_id

    # 4. Load keyframe pose and keypoints
    cursor.execute("SELECT id, pose_cw, undist_keypts, n_keypts FROM keyframes")
    kf_data = {}
    for kf_id, pose_blob, keypt_blob, n_keypts in cursor.fetchall():
        try:
            T_cw = np.frombuffer(pose_blob, dtype=np.float64).reshape(4, 4)
        except ValueError:
            T_cw = np.frombuffer(pose_blob, dtype=np.float32).reshape(4, 4)
        try:
            raw = np.frombuffer(keypt_blob, dtype=np.float32)
            keypoints = raw.reshape(n_keypts, 7)[:, :2]  # Only x, y coordinates
        except Exception:
            raise ValueError(
                f"❌ Failed to parse keypoints in kf {kf_id}: expected float32 × 7, got {len(keypt_blob)} bytes."
            )
        matched_3d = [
            landmarks.get(assoc_map.get((kf_id, i))) for i in range(n_keypts)
        ]
        # Assign image name to kf_id (if exists)
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
    Filters keyframe data, retaining only keypoints that have an associated 3D landmark.

    Args:
        kf_data (dict): image name → raw entry with fields:
            {
                "kf_id": int,
                "T_cw": np.ndarray (4x4),
                "keypoints": np.ndarray (N,2),
                "matched_3d": List[Optional[np.ndarray]]
            }

    Returns:
        Dict[str, Dict]: image name → filtered entry:
            {
                "kf_id": int,
                "T_cw": np.ndarray (4x4),
                "keypoints": List[Tuple[float, float]],
                "matched_3d": List[List[float]]
            }
        Only images with at least one valid match are included.
    """
    filtered: Dict[str, Dict[str, Any]] = {}
    for name, entry in kf_data.items():
        keypoints = entry.get("keypoints", [])
        matched_3d = entry.get("matched_3d", [])
        gp: List[Tuple[float, float]] = []
        lm: List[List[float]] = []

        # Keep only points with a valid 3D match
        for pt, m in zip(keypoints, matched_3d):
            if m is not None:
                x, y = float(pt[0]), float(pt[1])
                gp.append((x, y))
                lm.append(m.tolist() if isinstance(m, np.ndarray) else m)

        if gp:
            filtered[name] = {
                "kf_id":   entry["kf_id"],
                "T_cw":    entry["T_cw"],
                "keypoints": gp,
                "matched_3d": lm,
            }

    return filtered
