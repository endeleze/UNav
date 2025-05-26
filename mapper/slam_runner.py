import subprocess
import os
from config import UNavMappingConfig

def run_stella_vslam_dense(
    config: UNavMappingConfig
):
    # Get parameters from config
    slam_cfg = config.slam_config
    container_name = slam_cfg["container_name"]
    gpu_id = slam_cfg["gpu_id"]
    viewer = slam_cfg["viewer"]

    vocab = slam_cfg["vocab_path"]
    config_yaml = slam_cfg["config_yaml"]
    video = slam_cfg["video_path"]

    eval_log_dir = slam_cfg["eval_log_dir"]
    map_db_out = slam_cfg["map_db_out"]
    pc_out = slam_cfg["pc_out"]
    kf_out = slam_cfg["kf_out"]

    host_data_root = slam_cfg["host_data_root"]
    container_data_root = slam_cfg["container_data_root"]

    host_eval_log_dir = slam_cfg["host_eval_log_dir"]
    host_keyframe_dir = slam_cfg["host_keyframe_dir"]

    # Ensure output folders exist on host
    for path in [host_eval_log_dir, host_keyframe_dir]:
        os.makedirs(path, exist_ok=True)

    shell_command = (
        f'echo "[SLAM] Running for {config.floor}" && '
        f'./run_video_slam '
        f'-v "{vocab}" -c "{config_yaml}" -m "{video}" '
        f'--no-sleep --auto-term '
        f'--eval-log-dir "{eval_log_dir}" '
        f'--map-db-out "{map_db_out}" '
        f'--pc-out "{pc_out}" '
        f'--kf-out "{kf_out}" '
        f'{"--viewer none" if not viewer else ""} && '
        f'echo "[\u2713] SLAM done for {config.floor}" || echo "[X] SLAM failed for {config.floor}"'
    )

    cmd = [
        "docker", "run", "--rm", "-i",
        "--gpus", f"\"device={gpu_id}\"",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "--entrypoint", "bash",
        "-v", f"{host_data_root}:{container_data_root}",
        "-w", "/stella_vslam_examples/build",
        "--name", container_name,
        "stella_vslam_dense",
        "-c", shell_command
    ]

    print(f"[*] Launching SLAM: {config.floor} on GPU {gpu_id}")
    # Kill and remove existing container if any
    try:
        subprocess.run(["docker", "rm", "-f", container_name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[!] Removed existing container: {container_name}")
    except subprocess.CalledProcessError:
        pass

    subprocess.run(cmd)