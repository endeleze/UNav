"""
Main mapping pipeline for UNav system.
Performs visual SLAM, perspective slicing, feature extraction,
feature matching, and 3D triangulation.
"""

import sys
from config import UNavConfig

# ------------------- Configuration Section -------------------

# Allow passing arguments from shell:
# Usage: python main_mapping_pipeline.py <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>
if len(sys.argv) != 7:
    print(
        f"Usage: python {sys.argv[0]} <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>"
    )
    sys.exit(1)

DATA_TEMP_ROOT   = sys.argv[1]  # Temporary intermediate storage, e.g., '/mnt/data/UNav-IO/temp'
DATA_FINAL_ROOT  = sys.argv[2]  # Final output storage, e.g., '/mnt/data/UNav-IO/data'
FEATURE_MODEL    = sys.argv[3]  # Global descriptor model, e.g., 'DinoV2Salad'
PLACE            = sys.argv[4]  # Place/city name, e.g., 'New_York_City'
BUILDING         = sys.argv[5]  # Building name, e.g., 'LightHouse'
FLOOR            = sys.argv[6]  # Floor name, e.g., '3_floor'

# Initialize UNav mapping config
config = UNavConfig(
    data_temp_root=DATA_TEMP_ROOT,
    data_final_root=DATA_FINAL_ROOT,
    mapping_place=PLACE,
    mapping_building=BUILDING,
    mapping_floor=FLOOR,
    global_descriptor_model=FEATURE_MODEL
)

mapper_config = config.mapping_config
# ------------------- Pipeline Section -------------------

# 1. Run dense SLAM mapping
from mapper.slam_runner import run_stella_vslam_dense
run_stella_vslam_dense(mapper_config)

# 2. Slice equirectangular video into perspective images
from mapper.slicer import slice_perspectives
slicer_result = slice_perspectives(mapper_config)

# 3. Extract local and global features
from mapper.feature_extractor import extract_features_from_dir
extract_features_from_dir(mapper_config)

# 4. Match features and filter with geometric verification
from mapper.matcher import generate_and_stream_colmap
verified_pairs = generate_and_stream_colmap(mapper_config)

# 5. Triangulate 3D structure using COLMAP with known poses
from mapper.colmap_triangulator import run_colmap_triangulation
run_colmap_triangulation(mapper_config)
