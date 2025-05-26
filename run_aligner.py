"""
UNav Mapping: Floorplan-SLAM Alignment Pipeline

This script provides an interactive GUI for registering (aligning) a SLAM-reconstructed map
with a 2D floorplan. It is designed to allow the user to select corresponding points between
the SLAM point cloud (or camera trajectory) and the architectural floorplan,
then compute the transformation matrix between the two coordinate systems.

Typical workflow:
    1. Visualize SLAM keyframes/point cloud alongside the floorplan image.
    2. Select pairs of corresponding points in both views.
    3. Compute and export the optimal transformation matrix (e.g., affine or similarity).
    4. Save the transformation for downstream localization and navigation.

This is a preparatory step for accurate metric localization and path planning.
"""

import sys
from config import UNavConfig

# ------------------- Argument Parsing & Configuration -------------------

# Usage:
#   python main_mapping_pipeline.py <data_temp_root> <data_final_root> <place> <building> <floor>
if len(sys.argv) != 6:
    print(
        f"Usage: python {sys.argv[0]} <data_temp_root> <data_final_root> <place> <building> <floor>"
    )
    sys.exit(1)

DATA_TEMP_ROOT   = sys.argv[1]  # Path for temporary/intermediate files (e.g., SLAM outputs)
DATA_FINAL_ROOT  = sys.argv[2]  # Directory for final results (e.g., floorplan, matrix)
PLACE            = sys.argv[3]  # Name of the place/city/campus
BUILDING         = sys.argv[4]  # Name of the building
FLOOR            = sys.argv[5]  # Floor label/ID

# Initialize configuration (contains all relevant paths and metadata)
config = UNavConfig(
    data_temp_root=DATA_TEMP_ROOT,
    data_final_root=DATA_FINAL_ROOT,
    mapping_place=PLACE,
    mapping_building=BUILDING,
    mapping_floor=FLOOR
)

mapper_config = config.mapping_config

# ------------------- Floorplan-SLAM Alignment GUI -------------------

# Launch the Aligner GUI:
#   - The user will manually select correspondences between the SLAM map and floorplan.
#   - The system will compute and export the transformation matrix.
from mapper.aligner import run_aligner_gui
run_aligner_gui(mapper_config)

# Output: The resulting transformation matrix will be saved to the appropriate location
#         specified in the configuration for downstream localization/navigation modules.
