#!/usr/bin/env python3
import os
import cv2
import numpy as np
import re
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R
from config import UNavMappingConfig
from core.colmap.database_preparer import write_colmap_camera_and_images
from mapper.tools.slam.read_data import get_keyframe_image_list, read_keyframe_trajectory

def calculate_output_size(equirect_image_path: str, fov: float) -> Tuple[int, int]:
    """
    Calculate output image size for slicing, proportional to input width and FOV.
    Args:
        equirect_image_path (str): Path to equirectangular input image.
        fov (float): Horizontal field of view (degrees).
    Returns:
        (out_w, out_h): Output width and height (pixels), 16:9 aspect ratio.
    """
    img = cv2.imread(equirect_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(equirect_image_path)
    h, w = img.shape[:2]
    out_w = int((fov / 360.0) * w)
    out_h = int(out_w * 9 / 16)
    return out_w, out_h

def compute_yaw_pitch_from_Rcw(R_cw: np.ndarray, desired_yaw: float) -> Tuple[float, float]:
    """
    Compute required yaw and pitch in camera frame for a target world yaw.
    Args:
        R_cw (np.ndarray): 3x3 camera-to-world rotation matrix.
        desired_yaw (float): Yaw angle (degrees) in world coordinates.
    Returns:
        (yaw_deg, pitch_deg): Euler angles (degrees), in 'yx' order.
    """
    y_cam_world = R_cw[:, 1]
    z_down = np.array([0.0, 0.0, -1.0])

    axis = np.cross(y_cam_world, z_down)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        R_level = np.eye(3)
    else:
        axis /= norm
        angle = np.arccos(np.clip(np.dot(y_cam_world, z_down), -1.0, 1.0))
        R_level = R.from_rotvec(axis * angle).as_matrix()

    R_yaw_world = R.from_euler('y', desired_yaw, degrees=True).as_matrix()
    R_target = R_yaw_world @ R_level @ R_cw
    R_delta = R_cw.T @ R_target

    yaw_deg, pitch_deg = R.from_matrix(R_delta).as_euler('yx', degrees=True)
    return yaw_deg, pitch_deg

def equirectangular_to_perspective(
    equirect_img: np.ndarray,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    width: int,
    height: int,
    RADIUS: float = 128
) -> np.ndarray:
    """
    Slice a perspective view from an equirectangular image.
    Args:
        equirect_img (np.ndarray): Input equirectangular image (H x W x 3).
        fov_deg (float): Horizontal field of view (degrees).
        yaw_deg (float): Yaw angle (degrees, world Z).
        pitch_deg (float): Pitch angle (degrees, local Y).
        width (int): Output image width.
        height (int): Output image height.
        RADIUS (float): Projection sphere radius.
    Returns:
        np.ndarray: Perspective image (height, width, channels).
    """
    equirect_h, equirect_w = equirect_img.shape[:2]
    equ_cx = (equirect_w - 1) / 2.0
    equ_cy = (equirect_h - 1) / 2.0

    h_fov_deg = float(height) / width * fov_deg
    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0

    w_angle = (180 - fov_deg) / 2.0
    w_len = 2 * RADIUS * np.sin(np.radians(fov_deg / 2.0)) / np.sin(np.radians(w_angle))
    w_interval = w_len / (width - 1)

    h_angle = (180 - h_fov_deg) / 2.0
    h_len = 2 * RADIUS * np.sin(np.radians(h_fov_deg / 2.0)) / np.sin(np.radians(h_angle))
    h_interval = h_len / (height - 1)

    x_map = np.full((height, width), RADIUS, dtype=np.float32)
    y_range = (np.arange(width) - c_x) * w_interval
    z_range = -(np.arange(height) - c_y) * h_interval
    y_map = np.broadcast_to(y_range, (height, width))
    z_map = np.broadcast_to(z_range[:, np.newaxis], (height, width))

    D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
    xyz = np.stack((x_map, y_map, z_map), axis=-1) * (RADIUS / D)[..., np.newaxis]

    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    R_yaw, _ = cv2.Rodrigues(z_axis * np.radians(yaw_deg))
    R_pitch, _ = cv2.Rodrigues(np.dot(R_yaw, y_axis) * np.radians(-pitch_deg))

    xyz = xyz.reshape(-1, 3).T  # (3, H*W)
    xyz = R_yaw @ xyz
    xyz = R_pitch @ xyz
    xyz = xyz.T.reshape(height, width, 3)

    lon = np.arctan2(xyz[..., 1], xyz[..., 0])
    lat = np.arcsin(xyz[..., 2] / RADIUS)

    lon_map = lon / np.pi * 180 / 180 * equ_cx + equ_cx
    lat_map = -lat / np.pi * 180 / 90 * equ_cy + equ_cy

    persp = cv2.remap(
        equirect_img,
        lon_map.astype(np.float32),
        lat_map.astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_WRAP
    )

    return persp

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
    config: UNavMappingConfig
) -> List[Dict[str, Any]]:
    """
    Main slicing pipeline: generates perspective slices from 360 panoramas,
    saves them, and writes COLMAP meta files.
    Args:
        config (UNavMappingConfig): Config object with all paths and slicing params.
    Returns:
        List[Dict]: Metadata for each generated perspective slice.
    """
    slicer_config = config.slicer_config
    keyframe_dir = slicer_config["input_keyframe_dir"]
    traj_file = slicer_config["trajectory_file"]
    N = slicer_config["num_perspectives"]
    fov = slicer_config["fov"]
    pitch = slicer_config["pitch"]
    out_dir = slicer_config["output_perspective_dir"]
    os.makedirs(out_dir, exist_ok=True)

    kf_list = get_keyframe_image_list(keyframe_dir)
    poses = read_keyframe_trajectory(traj_file, kf_list)
    data: List[Dict[str, Any]] = []

    for name in tqdm(kf_list, desc="slicing"):
        img_path = os.path.join(keyframe_dir, name)
        pano = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if pano is None:
            print(f"[Skip] Cannot read image: {name}")
            continue
        pose4 = poses.get(name)
        if pose4 is None:
            print(f"[Skip] No pose for: {name}")
            continue

        R_cw = pose4[:3, :3]
        T_cw = pose4[:3, 3]
        R_wc = R_cw.T
        out_w, out_h = calculate_output_size(img_path, fov)

        for yaw_idx in range(N):
            yaw = (360.0 / N) * yaw_idx
            yaw_cam = R.from_euler('y', yaw, degrees=True).as_matrix()
            R_wc_slice = yaw_cam.T @ R_wc
            T_wc_slice = -R_wc_slice @ T_cw
            xyzw = R.from_matrix(R_wc_slice).as_quat()
            q_wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]

            out_name = generate_perspective_name(name, pitch, yaw_idx)
            out_path = os.path.join(out_dir, out_name)
            if os.path.exists(out_path):
                continue

            slice_img = equirectangular_to_perspective(
                pano, fov_deg=fov,
                yaw_deg=yaw,
                pitch_deg=pitch,
                width=out_w, height=out_h
            )
            cv2.imwrite(out_path, slice_img)

            data.append({
                "image_name": out_name,
                "image": None,
                "q_wxyz": q_wxyz,
                "t_c": T_wc_slice,
            })

    camcfg = config.colmap_config
    os.makedirs(os.path.dirname(camcfg['camera_file']), exist_ok=True)
    os.makedirs(os.path.dirname(camcfg['image_file']), exist_ok=True)
    write_colmap_camera_and_images(
        data, out_w, out_h, fov,
        camcfg['camera_file'],
        camcfg['image_file']
    )
    return data
