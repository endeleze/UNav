import tkinter as tk
from config import UNavMappingConfig

from mapper.tools.aligner.aligner_app import AlignerApp
from mapper.tools.aligner.aligner_logic import AlignerLogic
from mapper.tools.aligner.io_utils import load_scales
from mapper.tools.slam.read_data import (
    get_keyframe_image_list,
    read_keyframe_trajectory,
    extract_kf_pose_and_matches,
    filter_kf_data
)

def run_aligner_gui(config: UNavMappingConfig) -> None:
    """
    Launches the AlignerApp GUI for interactive floorplan/SLAM alignment,
    with all relevant keyframes and metadata preloaded.

    Args:
        config (UNavMappingConfig): Configuration object containing all paths,
            place/building/floor settings, and the aligner module's parameters.
    """
    # Extract configuration sections
    aligner_cfg = config.aligner_config
    slicer_config = config.slicer_config

    # Load SLAM keyframe pose/match data and filter unusable frames
    slam_output_path = aligner_cfg['map_db_out']
    keyframe_dir = slicer_config["input_keyframe_dir"]
    kf_data = extract_kf_pose_and_matches(slam_output_path, keyframe_dir)
    filtered_kf = filter_kf_data(kf_data)

    # Path to floorplan image
    floorplan_path = aligner_cfg['floorplan_path']

    # Load scale (meters per pixel) for the current place/building/floor
    scales = load_scales(aligner_cfg['scale_file'])
    scale = scales.get(config.place, {}).get(config.building, {}).get(config.floor, 1.0)

    # Load full SLAM trajectory (for all keyframes)
    traj_file = slicer_config["trajectory_file"]
    kf_list = get_keyframe_image_list(keyframe_dir)
    poses = read_keyframe_trajectory(traj_file, kf_list)

    # Update filtered_kf to use precise trajectory poses
    for img_name in filtered_kf:
        if img_name in poses:
            filtered_kf[img_name]["T_cw"] = poses[img_name]
        else:
            print(f"⚠️ Warning: {img_name} not found in poses, skipping T_cw replacement.")

    # Prepare alignment logic and launch the GUI
    logic = AlignerLogic(
        filtered_kf,
        keyframe_dir,
        floorplan_path,
        scale,
        config.aligner_config
    )

    root = tk.Tk()
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    app = AlignerApp(root, logic)
    app.grid(row=0, column=0, sticky="nsew")
    root.mainloop()
