#!/bin/bash
#
# UNav Mapping: Floorplan-SLAM Alignment Launcher
#
# This script launches the interactive alignment GUI to register a SLAM map
# with a 2D floorplan, enabling the user to compute the coordinate transformation matrix.
#
# Usage:
#   bash run_aligner.sh
#   (Modify PLACE, BUILDING, FLOOR, etc. below as needed)

# ------------------- User-Configurable Variables -------------------

DATA_TEMP_ROOT="/mnt/data/UNav-IO/temp"      # Temporary/intermediate files directory (e.g., SLAM outputs)
DATA_FINAL_ROOT="/mnt/data/UNav-IO/data"     # Final output directory (floorplan, transformation matrix, etc.)

PLACE="New_York_City"                        # Name of the place/city/campus
BUILDING="LightHouse"                        # Name of the building
FLOOR="4_floor"                              # Floor label/ID

# ------------------- Main Script Execution -------------------

# Launch the Python alignment pipeline
python run_aligner.py \
    "$DATA_TEMP_ROOT" \
    "$DATA_FINAL_ROOT" \
    "$PLACE" \
    "$BUILDING" \
    "$FLOOR"

# Note:
# - The GUI will open for manual alignment.
# - After you finish correspondence selection and alignment,
#   the transformation matrix will be saved automatically to the configured output path.
# - For batch processing, modify the variables above or loop over multiple floor/building names.
