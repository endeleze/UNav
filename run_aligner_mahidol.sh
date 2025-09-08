#!/bin/bash
#
# UNav Mapping: Floorplan-SLAM Alignment Launcher
#
# This script launches the interactive alignment GUI to register a SLAM map
# with a 2D floorplan, enabling the user to compute the coordinate transformation matrix.
#
# Usage:
#   bash run_aligner.sh
#   (Modify PLACE, BUILDING, FLOOR, etc. as needed)


# ------------------- User-Configurable Variables -------------------

#DATA_TEMP_ROOT="/mnt/data/UNav-IO/temp"      # Temporary/intermediate files directory (e.g., SLAM outputs)
#DATA_FINAL_ROOT="/mnt/data/UNav-IO/data"     # Final output directory (floorplan, transformation matrix, etc.)

DATA_TEMP_ROOT="/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data"
DATA_FINAL_ROOT="/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data"
#FEATURE_MODEL="NetVlad"
#FEATURE_MODEL="MixVPR"
#FEATURE_MODEL="DinoV2Salad"
#FEATURE_MODEL="CricaVPR"
FEATURE_MODEL="AnyLoc"

PLACE="Mahidol_University" # Name of the place/city/campus
BUILDING="ICT" # Name of the building
FLOOR="1.1_$FEATURE_MODEL" # Floor label/ID

# ------------------- Main Script Execution -------------------

echo "---------------------------------------------"
echo "Launching Floorplan-SLAM Aligner GUI:"
echo "  Place   : $PLACE"
echo "  Building: $BUILDING"
echo "  Floor   : $FLOOR"
echo "---------------------------------------------"

python3 -m unav.run_aligner \
    "$DATA_TEMP_ROOT" \
    "$DATA_FINAL_ROOT" \
    "$PLACE" \
    "$BUILDING" \
    "$FLOOR"

echo "✅ Floorplan-SLAM alignment completed. Transformation matrix saved."
echo ""
# Notes:
# - The GUI will open for manual alignment.
# - After you finish correspondence selection and alignment,
#   the transformation matrix will be saved automatically to the configured output path.
# - For batch processing, loop over PLACE/BUILDING/FLOOR variables above as needed.
