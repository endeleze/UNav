"""
UNav Floorplan-SLAM Alignment GUI Launcher

This module launches the interactive GUI for aligning a SLAM-reconstructed map
with a 2D architectural floorplan. It prepares all necessary data (keyframes,
trajectories, scale, floorplan image, etc.) and opens the GUI for correspondence
selection and transformation matrix computation.

Typical usage:
    from mapper.aligner import run_aligner_gui
    run_aligner_gui(mapping_config)
"""

import tkinter as tk
from unav.config import UNavMappingConfig

from unav.mapper.tools.aligner.aligner_app import AlignerApp
from unav.mapper.tools.aligner.aligner_logic import AlignerLogic
from unav.mapper.tools.aligner.io_utils import load_scales
from unav.mapper.tools.slam.read_data import (
    get_keyframe_image_list,
    read_keyframe_trajectory,
    extract_kf_pose_and_matches,
    filter_kf_data
)

def run_aligner_gui(config: UNavMappingConfig) -> None:
    """
    Launch the AlignerApp GUI for interactive floorplan/SLAM alignment.
    Loads all relevant keyframes, trajectories, and metadata in advance.

    Args:
        config (UNavMappingConfig): Configuration object containing all paths and parameters.
    """
    # Extract configuration sections
    aligner_cfg = config.aligner_config
    slicer_cfg = config.slicer_config

    # Load and filter SLAM keyframe pose/match data
    slam_output_path = aligner_cfg["map_db_out"]
    keyframe_dir = slicer_cfg["input_keyframe_dir"]
    kf_data = extract_kf_pose_and_matches(slam_output_path, keyframe_dir)
    filtered_kf = filter_kf_data(kf_data)

    # Floorplan image path
    floorplan_path = aligner_cfg["floorplan_path"]

    # Load pixel-to-metric scale (meters per pixel) for the current scene
    scales = load_scales(aligner_cfg["scale_file"])
    scale = scales.get(config.place, {}).get(config.building, {}).get(config.floor, 1.0)

    # Load SLAM trajectory (full 6-DoF pose for each keyframe)
    traj_file = slicer_cfg["trajectory_file"]
    kf_list = get_keyframe_image_list(keyframe_dir)
    poses = read_keyframe_trajectory(traj_file, kf_list)

    # Update filtered_kf dictionary with precise poses from the trajectory file
    for img_name in filtered_kf:
        if img_name in poses:
            filtered_kf[img_name]["T_cw"] = poses[img_name]
        else:
            print(f"⚠️ Warning: {img_name} not found in trajectory, using default pose.")

    # Prepare alignment logic and launch the GUI application
    logic = AlignerLogic(
        filtered_kf,
        keyframe_dir,
        floorplan_path,
        scale,
        aligner_cfg
    )

    root = tk.Tk()
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    app = AlignerApp(root, logic)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()
