#!/bin/bash
# UNav Mapping Batch Runner Script
# 
# This script batch-processes multiple places, buildings, and floors
# for the UNav mapping pipeline.
#
# Usage:
#   ./run_mapping_batch.sh
#
# Variables below can be modified for your own mapping jobs.


# ------------- User-Configurable Section -------------

DATA_TEMP_ROOT="/mnt/data/UNav-IO/temp"
DATA_FINAL_ROOT="/mnt/data/UNav-IO/data"
FEATURE_MODEL="DinoV2Salad"

PLACES=("New_York_University")
BUILDINGS=("Langone")
FLOORS=("15_floor" "16_floor")

# ------------- Main Batch Processing Loop ------------

for place in "${PLACES[@]}"; do
  for building in "${BUILDINGS[@]}"; do
    for floor in "${FLOORS[@]}"; do
      echo "---------------------------------------------"
      echo ">> Mapping: Place=$place | Building=$building | Floor=$floor"
      echo "---------------------------------------------"
      python -m unav.run_mapping \
        "$DATA_TEMP_ROOT" \
        "$DATA_FINAL_ROOT" \
        "$FEATURE_MODEL" \
        "$place" \
        "$building" \
        "$floor"
      echo ""
    done
  done
done

echo "✅ All mapping jobs finished successfully."
