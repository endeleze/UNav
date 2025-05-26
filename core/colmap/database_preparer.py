import logging
from pathlib import Path
import h5py
import numpy as np
from core.colmap.database import COLMAPDatabase
from core.colmap.read_write_model import CAMERA_MODEL_NAMES
from core.colmap.utils_pose import load_colmap_images_file_qt

def load_colmap_cameras_file(cameras_file: str) -> dict:
    """
    Load COLMAP standard cameras.txt.
    Returns:
        dict: {camera_id: {model_id, width, height, params}}
    """
    cameras = {}
    with open(cameras_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.strip() == '' or line.startswith('#'):
            continue
        tokens = line.strip().split()
        if len(tokens) < 5:
            continue

        camera_id = int(tokens[0])
        model_name = tokens[1]  # e.g. "PINHOLE"
        # ← 这里做映射：
        model_id  = CAMERA_MODEL_NAMES[model_name].model_id  
        width     = int(tokens[2])
        height    = int(tokens[3])
        params    = list(map(float, tokens[4:]))

        cameras[camera_id] = {
            'model_id': model_id,  # numeric, not text
            'width': width,
            'height': height,
            'params': params
        }

    logging.info(f"[UNav] Loaded {len(cameras)} cameras from {cameras_file}")
    return cameras

def create_colmap_database_with_known_poses(
    database_path: Path,
    local_feature_file: Path,
    matches_file: Path,
    cameras_txt: Path,
    images_txt: Path,
    pairs_txt: Path
):
    logging.info(f"[UNav] Creating COLMAP database with known poses from TXT files...")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    # --- Import camera
    cameras = load_colmap_cameras_file(cameras_txt)
    for cid, cam in cameras.items():
        db.add_camera(cam['model_id'], cam['width'], cam['height'], cam['params'], camera_id=cid, prior_focal_length=True)

    # --- Import images (known poses)
    poses = load_colmap_images_file_qt(images_txt)
    image_id_map = {}
    for name, data in poses.items():
        db.add_image(
            name, 
            camera_id=1,
            prior_q=data['qvec'], 
            prior_t=data['tvec'], 
            image_id=data['image_id']
        )
        image_id_map[name] = data['image_id']

    # --- Import keypoints
    with h5py.File(local_feature_file, 'r') as f_feat:
        for name in poses.keys():
            kpts = f_feat[name]['keypoints'][:]
            kpts = kpts.astype(np.float32) + 0.5  # COLMAP pixel‐center origin
            db.add_keypoints(image_id_map[name], kpts)

    # --- Import matches
    with open(pairs_txt, 'r') as f:
        pairs = [line.strip().split() for line in f.readlines() if line.strip()]

    with h5py.File(matches_file, 'r') as f_match:
        for img1_name, img2_name in pairs:
            if img1_name not in image_id_map or img2_name not in image_id_map:
                logging.error(f"[UNav] Unknown image in pairs.txt: {img1_name} or {img2_name}")
                continue

            img1_id = image_id_map[img1_name]
            img2_id = image_id_map[img2_name]

            # 使用标准拼接
            key = f"{img1_name}_{img2_name}"
            if key not in f_match:
                key = f"{img2_name}_{img1_name}"
                if key not in f_match:
                    logging.error(f"[UNav] Could not find matches for pair {img1_name} {img2_name}")
                    continue

            matches = f_match[key]['matches0'][:]
            db.add_matches(img1_id, img2_id, matches)
            db.add_two_view_geometry(
                img1_id,
                img2_id,
                matches,
                F=np.eye(3),
                E=np.eye(3),
                H=np.eye(3),
                config=2
            )
    db.commit()
    db.close()
    logging.info(f"[UNav] COLMAP database created and populated.")