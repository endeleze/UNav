import subprocess
import os
from unav.config import UNavMappingConfig

def run_stella_vslam_dense(
    config: UNavMappingConfig
):
    """
    Launches the stella_vslam_dense SLAM system inside a Docker container.
    All parameters are provided via the configuration object.
    The function will ensure required output directories exist and manage the Docker container lifecycle.

    Args:
        config (UNavMappingConfig): Unified config object containing all SLAM and mapping parameters.
    """
    # Extract all SLAM-specific parameters from the config
    slam_cfg = config.slam_config
    container_name = slam_cfg["container_name"]
    gpu_id         = slam_cfg["gpu_id"]
    viewer         = slam_cfg["viewer"]

    vocab          = slam_cfg["vocab_path"]
    config_yaml    = slam_cfg["config_yaml"]
    video          = slam_cfg["video_path"]

    eval_log_dir   = slam_cfg["eval_log_dir"]
    map_db_out     = slam_cfg["map_db_out"]
    pc_out         = slam_cfg["pc_out"]
    kf_out         = slam_cfg["kf_out"]

    host_data_root      = slam_cfg["host_data_root"]
    container_data_root = slam_cfg["container_data_root"]
    host_eval_log_dir   = slam_cfg["host_eval_log_dir"]
    host_keyframe_dir   = slam_cfg["host_keyframe_dir"]

    # Ensure that host output directories exist
    for path in [host_eval_log_dir, host_keyframe_dir]:
        os.makedirs(path, exist_ok=True)

    # Compose the shell command that will be executed in the Docker container
    shell_command = (
        f'echo "[SLAM] Running for {config.floor}" && '
        f'./run_video_slam '
        f'-v "{vocab}" -c "{config_yaml}" -m "{video}" '
        f'--no-sleep --auto-term '
        f'--eval-log-dir "{eval_log_dir}" '
        f'--map-db-out "{map_db_out}" '
        f'--pc-out "{pc_out}" '
        f'--kf-out "{kf_out}" '
        f'{"--viewer none" if not viewer else ""}'
    )

    # Build the Docker command with correct GPU assignment and mounts
    mem_limit = slam_cfg.get("mem_limit")  # e.g. "90g"; None disables the cap
    cmd = [
        "docker", "run", "--rm", "-i",
        "--gpus", f"device={gpu_id}",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
    ]
    if mem_limit:
        # --memory-swap == --memory disables container swap so it fails fast/cleanly.
        cmd += ["--memory", str(mem_limit), "--memory-swap", str(mem_limit)]
    cmd += [
        "--entrypoint", "bash",
        "-v", f"{host_data_root}:{container_data_root}",
        "-w", "/stella_vslam_examples/build",
        "--name", container_name,
        "stella_vslam_dense",
        "-c", shell_command
    ]

    print(f"[*] Launching SLAM for floor '{config.floor}' on GPU {gpu_id}")

    # If the container already exists, kill and remove it
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"[!] Removed existing container: {container_name}")
    except subprocess.CalledProcessError:
        # Ignore errors if container did not exist
        pass

    # Launch SLAM in the container
    result = subprocess.run(cmd)

    # Fail fast: SLAM must exit cleanly AND write its keyframe trajectory. The most
    # common failure is OOM (dense SLAM grows to tens of GB), which SIGKILLs
    # run_video_slam (exit 137) before the trajectory is written. Abort here with a
    # clear message instead of letting the downstream slicer crash on a missing file.
    traj_file = os.path.join(host_eval_log_dir, "keyframe_trajectory.txt")
    if result.returncode != 0 or not os.path.exists(traj_file):
        raise RuntimeError(
            f"[SLAM] dense SLAM FAILED for {config.floor} "
            f"(docker exit={result.returncode}, trajectory_written={os.path.exists(traj_file)}). "
            f"Most likely OOM: run_video_slam can grow to tens of GB. Run one floor at a "
            f"time, lower the video bitrate/resolution (or use run_mapper_segment), and/or "
            f"lower slam_config['mem_limit']. Aborting before slicing."
        )
    print(f"[✓] SLAM done for {config.floor}")
