# UNav Localizer Module

This folder provides the **modular, scalable localization pipeline** for the UNav navigation system—enabling robust visual localization across multi-building, multi-floor indoor environments.

All core steps are reusable Python APIs, with a unified pipeline and fine-grained control for both research and deployment.

---

## 📦 Prerequisites

Please install **all dependencies** before using the localizer:

- **Mapping outputs:** Run the full UNav Mapping pipeline first to generate all required features, COLMAP models, and transform matrices.
- **CUDA GPU:** Strongly recommended for real-time performance.
- **Python libraries:**  
    - [implicit_dist](https://github.com/cvg/implicit_dist.git)
    - [PoseLib](https://github.com/vlarsson/PoseLib)
    - (For mapping) [stella_vslam_dense Docker](https://github.com/RoblabWh/stella_vslam_dense.git)
- Example installation:
    ```bash
    git clone https://github.com/cvg/implicit_dist.git
    cd implicit_dist
    pip install .

    git clone https://github.com/vlarsson/PoseLib.git
    cd PoseLib
    pip install .
    ```

---

## 🚀 Pipeline Overview

The UNav Localizer operates as follows:

1. **Feature Extraction:** Extract global & local features from the query image.
2. **VPR Retrieval:** Retrieve Top-K most similar database images across all registered places/floors.
3. **Data Preparation:** Load candidate map features and 3D geometry.
4. **Local Matching & RANSAC:** Batch geometric verification for all candidates, filtering outliers.
5. **Multi-frame Pose Refinement:** Optional—jointly refine pose with historical matches.
6. **Floorplan Projection:** Transform the refined 6DoF pose to 2D floorplan coordinates.

---

## 🚦 Example Usage

```python
import cv2
from config import UNavConfig
from localizer.localizer import UNavLocalizer

# ---- 1. Build config and initialize ----
DATA_FINAL_ROOT = "/mnt/data/UNav-IO/data"
PLACES    = ["New_York_City"]
BUILDINGS = ["LightHouse"]
FLOORS    = ["3_floor", "4_floor", "6_floor"]

config = UNavConfig(
    data_final_root=DATA_FINAL_ROOT,
    places=PLACES,
    buildings=BUILDINGS,
    floors=FLOORS,
    global_descriptor_model="DinoV2Salad",
    local_feature_model="superpoint+lightglue"
)
localizer_config = config.localizer_config

localizer = UNavLocalizer(localizer_config)
localizer.load_maps_and_features()

# ---- 2. Query image localization ----
img = cv2.imread("/mnt/data/UNav-IO/logs/New_York_City/LightHouse/6/02754/images/2023-07-18_09-40-46.png")

# Extract features
global_feat, local_feat_dict = localizer.extract_query_features(img)

# VPR retrieval
top_candidates = localizer.vpr_retrieve(global_feat, top_k=50)

# Candidate data preparation
candidates_data = localizer.get_candidates_data(top_candidates)

# Batch local matching + RANSAC
best_map_key, pnp_pairs, results = localizer.batch_local_matching_and_ransac(local_feat_dict, candidates_data)

# Multi-frame pose refinement (optional)
refinement_queue = {}
map_queue = refinement_queue.get(best_map_key, {"pairs": [], "initial_poses": [], "pps": []})
refine_result = localizer.multi_frame_pose_refine(pnp_pairs, img.shape, map_queue)

# Floorplan projection
colmap_pose = {"qvec": refine_result.get("qvec"), "tvec": refine_result.get("tvec")}
transform_matrix = localizer.transform_matrices.get(best_map_key, None)
floorplan_pose = (
    localizer.transform_pose_to_floorplan(colmap_pose["qvec"], colmap_pose["tvec"], transform_matrix)
    if (colmap_pose["tvec"] is not None and transform_matrix is not None)
    else None
)
```

---

## 🧩 Modular API Reference

Each pipeline stage is accessible as a standalone API:
```python
localizer.extract_query_features(query_img)
localizer.vpr_retrieve(global_feat, top_k)
localizer.get_candidates_data(top_candidates)
localizer.batch_local_matching_and_ransac(local_feat_dict, candidates_data)
localizer.multi_frame_pose_refine(pnp_pairs, img_shape, queue)
localizer.transform_pose_to_floorplan(qvec, tvec, transform_matrix)
```
Or use the **unified one-shot call**:
```python
result = localizer.localize(query_img, refinement_queue, top_k=...)
```

---

## ⚙️ Example Configuration (`config.py`)

```python
from config import UNavConfig

config = UNavConfig(
    data_final_root="/mnt/data/UNav-IO/data",
    places=["New_York_City"],
    buildings=["LightHouse"],
    floors=["3_floor", "4_floor", "6_floor"],
    global_descriptor_model="DinoV2Salad",
    local_feature_model="superpoint+lightglue"
)
```

---

## 📂 Directory Structure

```text
localizer/
│
├── localizer.py
├── tools/
│   ├── io.py
│   ├── feature_extractor.py
│   ├── retriever.py
│   ├── matcher.py
│   └── pnp.py
├── visualization_tools/
│   └── localization_visualization_tools.py
├── config.py
└── README.md
```

---

## 🎨 Visualization
- To simulate and visually debug the localization process, use the Jupyter notebook:

    ```
    /home/unav/Desktop/unav/visualization_localization.ipynb
    ```

  This notebook provides step-by-step demonstration and diagnostic tools for end-to-end visual localization evaluation.

---

## 💡 Best Practices

- Call `load_maps_and_features()` before the first query, or after data changes.
- Use a **GPU** for all heavy computation steps (matching, RANSAC, refinement).
- For robust navigation, **multi-frame pose refinement** is recommended.
- The system supports seamless multi-building/multi-floor expansion with a single configuration.

---

## 👤 Contact

Developer: Anbang Yang (`ay1620@nyu.edu`)

_Last updated: 2025-05-27_
