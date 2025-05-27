#!/bin/bash

DATA_TEMP_ROOT="/mnt/data/UNav-IO/temp"
DATA_FINAL_ROOT="/mnt/data/UNav-IO/data"
FEATURE_MODEL="DinoV2Salad"

PLACES=("New_York_City")
BUILDINGS=("LightHouse")
FLOORS=("6_floor")

for place in "${PLACES[@]}"; do
  for building in "${BUILDINGS[@]}"; do
    for floor in "${FLOORS[@]}"; do
      echo "Processing $place - $building - $floor"
      python main_mapping_pipeline.py "$DATA_TEMP_ROOT" "$DATA_FINAL_ROOT" "$FEATURE_MODEL" "$place" "$building" "$floor"
    done
  done
done