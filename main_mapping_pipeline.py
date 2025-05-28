#!/usr/bin/env python3
"""
UNav Main Mapping Pipeline

This script orchestrates the mapping process for the UNav system,
including the following core stages:
    1. Visual SLAM for dense mapping
    2. Equirectangular-to-perspective slicing
    3. Feature extraction (local & global)
    4. Feature matching with geometric verification
    5. 3D triangulation with COLMAP

Typical usage:
    python main_mapping_pipeline.py <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>
"""

import sys
from unav.config import UNavConfig

def parse_args() -> tuple:
    """
    Parse command-line arguments.

    Returns:
        Tuple containing:
            data_temp_root (str)
            data_final_root (str)
            feature_model (str)
            place (str)
            building (str)
            floor (str)
    """
    if len(sys.argv) != 7:
        print(
            f"Usage: python {sys.argv[0]} "
            "<data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>"
        )
        sys.exit(1)
    return tuple(sys.argv[1:])

def main():
    """
    Main function for running the UNav mapping pipeline.
    """
    # ------------------- Configuration Section -------------------
    (
        data_temp_root,
        data_final_root,
        feature_model,
        place,
        building,
        floor
    ) = parse_args()

    # Initialize global config
    config = UNavConfig(
        data_temp_root=data_temp_root,
        data_final_root=data_final_root,
        mapping_place=place,
        mapping_building=building,
        mapping_floor=floor,
        global_descriptor_model=feature_model
    )
    mapper_config = config.mapping_config

    # ------------------- Pipeline Section -------------------

    # 1. Run dense SLAM mapping
    from unav.mapper.slam_runner import run_stella_vslam_dense
    run_stella_vslam_dense(mapper_config)

    # 2. Slice equirectangular video into perspective images
    from unav.mapper.slicer import slice_perspectives
    slicer_result = slice_perspectives(mapper_config)

    # 3. Extract local and global features
    from unav.mapper.feature_extractor import extract_features_from_dir
    extract_features_from_dir(mapper_config)

    # 4. Match features and filter with geometric verification
    from unav.mapper.matcher import generate_and_stream_colmap
    verified_pairs = generate_and_stream_colmap(mapper_config)

    # 5. Triangulate 3D structure using COLMAP with known poses
    from unav.mapper.colmap_triangulator import run_colmap_triangulation
    run_colmap_triangulation(mapper_config)

if __name__ == "__main__":
    main()
