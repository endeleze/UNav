import logging
from pathlib import Path
from subprocess import run
from config import UNavMappingConfig
from core.colmap.database_preparer import create_colmap_database_with_known_poses
import shutil

def run_colmap_triangulation(
    config: UNavMappingConfig,
    overwrite_database: bool = True,
    overwrite_output: bool = True
) -> None:
    """
    Run the complete COLMAP triangulation pipeline:
      1. Prepare the COLMAP database with known camera poses and matches.
      2. Optionally remove existing database/output directories if overwrite is enabled.
      3. Run COLMAP's point_triangulator with all necessary paths.

    Args:
        config (UNavMappingConfig): Configuration object containing all required paths.
        overwrite_database (bool): Remove existing COLMAP database before creation if True.
        overwrite_output (bool): Remove existing triangulated output before execution if True.

    Raises:
        RuntimeError: If the COLMAP point_triangulator process fails.
    """
    # === Resolve all paths from config ===
    colmap_cfg = config.colmap_config
    feat_cfg = config.feature_extraction_config

    database_path = Path(colmap_cfg["database_path"])
    local_feature_file = Path(feat_cfg["local_feat_save_path"])
    matches_file = Path(colmap_cfg["match_file"])
    cameras_txt = Path(colmap_cfg["camera_file"])
    images_txt = Path(colmap_cfg["image_file"])
    pairs_txt = Path(colmap_cfg["pairs_txt"])
    image_dir = Path(feat_cfg["input_perspective_dir"])
    sparse_model_dir = Path(colmap_cfg["sparse_dir"])
    output_dir = Path(colmap_cfg["colmap_output_dir"])

    # === Handle overwrite options ===
    if database_path.exists() and overwrite_database:
        logging.warning(f"[UNav] Overwriting existing COLMAP database at: {database_path}")
        database_path.unlink()

    if output_dir.exists() and overwrite_output:
        logging.warning(f"[UNav] Overwriting existing triangulation output at: {output_dir}")
        shutil.rmtree(output_dir)

    # === Create COLMAP database from features, matches, and known poses ===
    logging.info(f"[UNav] Creating COLMAP database at: {database_path}")
    create_colmap_database_with_known_poses(
        database_path=database_path,
        local_feature_file=local_feature_file,
        matches_file=matches_file,
        cameras_txt=cameras_txt,
        images_txt=images_txt,
        pairs_txt=pairs_txt
    )

    # Ensure sparse_model_dir/points3D.txt exists (COLMAP expects it)
    points3d_txt = sparse_model_dir / "points3D.txt"
    if not points3d_txt.exists():
        logging.warning(f"[UNav] points3D.txt not found in {sparse_model_dir}, creating empty placeholder file.")
        points3d_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(points3d_txt, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")

    # === Run COLMAP triangulation ===
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "colmap", "point_triangulator",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--input_path", str(sparse_model_dir),
        "--output_path", str(output_dir),
    ]

    logging.info(f"[UNav] Launching COLMAP: {' '.join(cmd)}")
    try:
        run(cmd, check=True)
    except Exception as e:
        logging.error(f"[UNav] COLMAP point_triangulator failed: {e}")
        raise RuntimeError("COLMAP point_triangulator execution failed.") from e

    logging.info(f"[UNav] Triangulation completed at: {output_dir}")
