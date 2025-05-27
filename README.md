# 🗺️ UNav: Unified Visual Navigation System

A **modular, scalable visual navigation framework** for large buildings, supporting mapping, localization, and pathfinding across multiple floors and buildings. Designed for real-world deployment in robotics, assistive navigation, and digital twin scenarios.

---

## 📦 Prerequisites

Before using UNav, ensure the following are installed:

- **Python 3.8+** (`requirements.txt`)
- **CUDA-compatible GPU** (recommended for feature extraction/matching)
- **Docker** (required for SLAM mapping with stella_vslam_dense)
- **COLMAP** (for triangulation; should be on your `$PATH`)
- **labelme** (for annotating floorplans)
- **[stella_vslam_dense](https://github.com/RoblabWh/stella_vslam_dense.git)** (SLAM mapping via Docker)
- **[implicit_dist](https://github.com/cvg/implicit_dist.git)** (multi-frame pose refinement)
- **[PoseLib](https://github.com/vlarsson/PoseLib)** (robust pose estimation)

**Quick setup for SLAM:**
```sh
git clone https://github.com/RoblabWh/stella_vslam_dense.git
cd stella_vslam_dense/docker
docker build -t stella_vslam_dense .
```

---

## 🚀 Getting Started (Mapping Pipeline Only)

### 1. One-command Installation

Install the UNav system (including all Python dependencies) via pip:

```sh
pip install git+https://github.com/ai4ce/unav.git
```

### 2. Run the Mapping Pipeline

This runs the full mapping pipeline (dense SLAM, slicing, feature extraction, matching, triangulation):

```sh
python main_mapping_pipeline.py <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>
```

Example:
```sh
python main_mapping_pipeline.py /mnt/data/UNav-IO/temp /mnt/data/UNav-IO/data DinoV2Salad New_York_City LightHouse 4_floor
```

### 3. Align 3D Map to Floorplan (Optional but Recommended)

Run the floorplan alignment GUI for metric registration:

```sh
python mapper/aligner.py <data_temp_root> <data_final_root> <place> <building> <floor>
```

---

## 📝 Example Notebooks

- `visualize_mapping.ipynb`  
  Visualize mapping results: point clouds, camera trajectories, feature quality.
- `visualize_localization.ipynb`  
  Inspect localization performance, candidate matches, and pose transformation.
- `visualize_navigation.ipynb`  
  Simulate navigation, visualize multi-floor routes, and review generated commands.

---

## 📖 Full Documentation & Code

For further details, localization, navigation modules, and developer guides, visit:  
[https://github.com/ai4ce/unav](https://github.com/ai4ce/unav)

---

## 👤 Maintainer

- **Developer:** Anbang Yang (`ay1620@nyu.edu`)
- **Last updated:** 2025-05-27

