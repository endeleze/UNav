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

usage(){
    printf "Usage: %s: -a <Algorithm> -p <Place> -b <Building> -f <Floor>\n" $(basename $0) >&2
    exit 2
}

while getopts 'a:p:b:f:' OPTION "$@"
do
    case $OPTION in
    a)  FEATURE_MODEL=$OPTARG
        ;;
    p)  PLACE=$OPTARG
        ;;
    b)  BUILDING=$OPTARG
        ;;
    f)  FLOOR=$OPTARG
        ;;
    ?)  usage
        ;;
    *)  echo "Nothing"
        usage
    esac
done

# ------------- User-Configurable Section -------------

DATA_TEMP_ROOT="/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data"
DATA_FINAL_ROOT="/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data"
# FEATURE_MODEL="MixVPR"

# PLACES=("Mahidol_University")
# BUILDINGS=("ICT")
# FLOORS=("1.1")

PLACES=$PLACE
BUILDINGS=$BUILDING
FLOORS=$FLOOR

# ------------- Main Batch Processing Loop ------------

for place in "${PLACES[@]}"; do
  for building in "${BUILDINGS[@]}"; do
    for floor in "${FLOORS[@]}"; do
      echo "---------------------------------------------"
      echo ">> Mapping: Place=$place | Building=$building | Floor=$floor"
      echo "---------------------------------------------"
      python3 -m unav.run_mapping \
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
