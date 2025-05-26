import logging
from pathlib import Path
from subprocess import run
from config import UNavMappingConfig

from core.colmap.database_preparer import create_colmap_database_with_known_poses


def run_colmap_triangulation(
    config: UNavMappingConfig,
    overwrite_database: bool = True,
    overwrite_output: bool = True
) -> None:
    """
    Full pipeline to prepare COLMAP database from H5 files and known poses in TXT,
    then run triangulation using known poses.

    Args:
        config: UNavMappingConfig object containing all paths and settings.
        overwrite_database: If True, existing database will be removed before creation.
        overwrite_output: If True, existing triangulation output will be removed before execution.

    Raises:
        RuntimeError: If COLMAP point_triangulator fails.
    """
    # --- Resolve paths from config
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

    # --- Safety checks and overwrite behavior
    if database_path.exists() and overwrite_database:
        logging.warning(f"[UNav] Overwriting existing database: {database_path}")
        database_path.unlink()

    if output_dir.exists() and overwrite_output:
        logging.warning(f"[UNav] Overwriting existing triangulated output: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)

    # --- Create COLMAP database
    logging.info(f"[UNav] Creating COLMAP database at {database_path}")
    create_colmap_database_with_known_poses(
        database_path=database_path,
        local_feature_file=local_feature_file,
        matches_file=matches_file,
        cameras_txt=cameras_txt,
        images_txt=images_txt,
        pairs_txt=pairs_txt
    )
    
    points3d_txt = sparse_model_dir / "points3D.txt"
    if not points3d_txt.exists():
        logging.warning(f"[UNav] points3D.txt not found in {sparse_model_dir}, creating empty file.")
        with open(points3d_txt, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")

    # --- Run triangulation
    output_dir.mkdir(exist_ok=True, parents=True)
    cmd = [
        "colmap", "point_triangulator",
        "--database_path",            str(database_path),
        "--image_path",               str(image_dir),
        "--input_path",               str(sparse_model_dir),
        "--output_path",              str(output_dir),
    ]


    logging.info(f"[UNav] CMD ➔ {' '.join(cmd)}")

    logging.info(f"[UNav] Running COLMAP point_triangulator...")
    try:
        run(cmd, check=True)
    except Exception as e:
        logging.error(f"[UNav] COLMAP point_triangulator failed: {e}")
        raise RuntimeError("COLMAP point_triangulator execution failed.") from e

    logging.info(f"[UNav] Triangulation completed at {output_dir}")
