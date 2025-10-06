import os, sys
import cv2
import time
import numpy as np
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from unav.config import UNavConfig
# import run_mapping as rmap
from unav.config import UNavMappingConfig
from unav.config import UNavConfig
from unav.mapper import slicer
from scipy.spatial.transform import Rotation as R

def generate_perspective_name(keyframe_name: str, pitch_idx: int, yaw_idx: int) -> str:
    """
    Generate standardized perspective image name.
    Args:
        keyframe_name (str): Base keyframe image name.
        pitch_idx (int): Pitch index.
        yaw_idx (int): Yaw index.
    Returns:
        str: Perspective image file name.
    """
    idx = int(re.findall(r'\d+', keyframe_name)[0])
    return f"{idx:06d}_pitch{pitch_idx:02d}_yaw{yaw_idx:02d}.png"

def slice_perspectives(
    config: UNavMappingConfig,
    img_dir: str
) -> List[Dict[str, Any]]:
    """
    Main slicing pipeline: generates perspective slices from 360 panoramas,
    saves them, and writes COLMAP meta files.
    Args:
        config (UNavMappingConfig): Config object with all paths and slicing params.
    Returns:
        List[Dict]: Metadata for each generated perspective slice.
    """
    out_dir = f'{img_dir}/perspective'
    slicer_config = config.slicer_config
    # keyframe_dir = slicer_config["input_keyframe_dir"]
    # traj_file = slicer_config["trajectory_file"]
    N = slicer_config["num_perspectives"]
    fov = slicer_config["fov"]
    pitch = slicer_config["pitch"]
    # out_dir = slicer_config["output_perspective_dir"]
    if os.path.exists(out_dir):
        print(f"[INFO] Output dir exists. Removing: {out_dir}")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # kf_list = get_keyframe_image_list(keyframe_dir)
    kf_list = os.listdir(img_dir)
    # poses = read_keyframe_trajectory(traj_file, kf_list)
    data: List[Dict[str, Any]] = []

    for name in tqdm(kf_list, desc="slicing"):
        img_path = os.path.join(img_dir, name)
        pano = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if pano is None:
            print(f"[Skip] Cannot read image: {name}")
            continue
        # pose4 = poses.get(name)
        # if pose4 is None:
        #     print(f"[Skip] No pose for: {name}")
        #     continue

        # R_cw = pose4[:3, :3]
        # T_cw = pose4[:3, 3]
        # R_wc = R_cw.T
        out_w, out_h = slicer.calculate_output_size(img_path, fov)

        for yaw_idx in range(N):
            yaw = (360.0 / N) * yaw_idx
            yaw_cam = R.from_euler('y', yaw, degrees=True).as_matrix()
            # R_wc_slice = yaw_cam.T @ R_wc
            # T_wc_slice = -R_wc_slice @ T_cw
            # xyzw = R.from_matrix(R_wc_slice).as_quat()
            # q_wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]

            out_name = slicer.generate_perspective_name(name, pitch, yaw_idx)
            out_path = os.path.join(out_dir, out_name)

            slice_img = slicer.equirectangular_to_perspective(
                pano, fov_deg=fov,
                yaw_deg=yaw,
                pitch_deg=pitch,
                width=out_w, height=out_h
            )
            cv2.imwrite(out_path, slice_img)

            data.append({
                "image_name": out_name,
                "image": None,
                # "q_wxyz": q_wxyz,
                # "t_c": T_wc_slice,
            })

    # camcfg = config.colmap_config
    # os.makedirs(os.path.dirname(camcfg['camera_file']), exist_ok=True)
    # os.makedirs(os.path.dirname(camcfg['image_file']), exist_ok=True)
    # write_colmap_camera_and_images(
    #     data, out_w, out_h, fov,
    #     camcfg['camera_file'],
    #     camcfg['image_file']
    # )
    return data

def parse_args() -> tuple:
    if len(sys.argv) != 2:
        print(
            f"Usage: python {sys.argv[0]} "
            "<images_dir_path>"
        )
        sys.exit(1)
    return tuple(sys.argv[1:])

def main():
    # ------------------- Configuration Section -------------------
    (
        img_path
    ) = parse_args()

    # Initialize global config
    config = UNavConfig(data_temp_root='dump', generate_stella_yaml=False)
    mapper_config = config.mapping_config

    # ------------------- Pipeline Section -------------------
    t0 = time.time()
    print("Starting slicing into perspective images ...")
    print('===========', img_path)
    slicer_result = slice_perspectives(mapper_config, img_path[0])
    print(f"Perspective slicing completed in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()