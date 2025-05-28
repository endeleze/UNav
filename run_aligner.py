#!/usr/bin/env python3
"""
UNav Mapping: Floorplan-SLAM Alignment Pipeline

This script launches an interactive GUI for registering (aligning)
a SLAM-reconstructed map with a 2D architectural floorplan.

Workflow steps:
    1. Visualize SLAM keyframes/point cloud alongside the floorplan image.
    2. Manually select pairs of corresponding points in both views.
    3. Compute the optimal transformation matrix (e.g., affine or similarity).
    4. Export the transformation for downstream localization/navigation.

This step is required for precise metric localization and reliable path planning.
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
            place (str)
            building (str)
            floor (str)
    """
    if len(sys.argv) != 6:
        print(
            f"Usage: python {sys.argv[0]} <data_temp_root> <data_final_root> <place> <building> <floor>"
        )
        sys.exit(1)
    return tuple(sys.argv[1:])

def main():
    """
    Main function to launch the floorplan-SLAM alignment GUI.
    """
    (
        data_temp_root,
        data_final_root,
        place,
        building,
        floor
    ) = parse_args()

    config = UNavConfig(
        data_temp_root=data_temp_root,
        data_final_root=data_final_root,
        mapping_place=place,
        mapping_building=building,
        mapping_floor=floor
    )
    mapper_config = config.mapping_config

    # Launch the Aligner GUI for manual correspondence and transformation computation
    from unav.mapper.aligner import run_aligner_gui
    run_aligner_gui(mapper_config)

    # Output: The transformation matrix will be saved for use in localization/navigation

if __name__ == "__main__":
    main()
