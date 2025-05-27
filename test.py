import cv2
from config import UNavConfig

DATA_FINAL_ROOT = "/mnt/data/UNav-IO/data"
FEATURE_MODEL = "DinoV2Salad"
LOCAL_FEATURE_MODEL = "superpoint+lightglue"

PLACES = ["New_York_City"]
BUILDINGS = ["LightHouse"]
FLOORS = ["3_floor", "4_floor"]

# Step 1: Build config and initialize localizer
config = UNavConfig(
    data_final_root=DATA_FINAL_ROOT,
    places=PLACES,
    buildings=BUILDINGS,
    floors=FLOORS,
    global_descriptor_model=FEATURE_MODEL,
    local_feature_model=LOCAL_FEATURE_MODEL
)
localizor_config = config.localizer_config

from localizer.localizer import UNavLocalizer

localizer = UNavLocalizer(localizor_config)
localizer.load_maps_and_features()

# Step 2: Load test image
test_image_path = "/mnt/data/UNav-IO/logs/New_York_City/LightHouse/3_floor/06465/images/2024-04-16_14-58-11.png"
img = cv2.imread(test_image_path)
if img is None:
    raise FileNotFoundError(f"Cannot load test image: {test_image_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 3: Initialize refinement_queue as a map dict (empty for all maps)
refinement_queue = {}
# 可选：你也可以只初始化你要用的map/floor
# for place in PLACES:
#     for building in BUILDINGS:
#         for floor in FLOORS:
#             map_key = f"{place}__{building}__{floor}"
#             refinement_queue[map_key] = {
#                 "pairs": [], "initial_poses": [], "pps": []
#             }

# Step 4: Run localization
result = localizer.localize(
    query_img=img_rgb,
    refinement_queue=refinement_queue,  # 支持多 map_key
    top_k=50
)

# Step 5: Print/inspect result
import pprint
pprint.pprint(result)

# Step 6: 若需要将 refinement_queue 更新为下一帧输入，可直接用 result["refinement_queue"]
refinement_queue = result["refinement_queue"]

# Step 7: 连续多帧循环调用时：
#   result = localizer.localize(next_img, refinement_queue=refinement_queue, ...)
