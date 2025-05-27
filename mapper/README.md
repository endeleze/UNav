# UNav Mapping Module

A complete, modular **mapping pipeline** and **floorplan-SLAM alignment toolkit** for the UNav visual navigation system.

---

## 📦 Prerequisites

Before running the mapping pipeline, **please make sure the following dependencies are installed and properly set up**:

- **Python 3.8+** with all dependencies listed in your `requirements.txt`.
- **Docker** is required for SLAM:
    - The **stella_vslam_dense** container must be built and available.  
      See official repository:  
      https://github.com/RoblabWh/stella_vslam_dense.git

    **Quick install:**
    ```sh
    git clone https://github.com/RoblabWh/stella_vslam_dense.git
    cd stella_vslam_dense/docker
    docker build -t stella_vslam_dense .
    ```

- **COLMAP** must be installed and accessible from your `$PATH` for triangulation.

- **CUDA-compatible GPU** is strongly recommended for feature extraction and matching.

---

## Overview

This folder contains all scripts and tools to convert raw sensor data into a metrically registered map suitable for robust visual localization and navigation, particularly in multi-floor and large-scale indoor environments.

- **Fully modularized:** Each pipeline stage is a self-contained Python module.
- **Batch and GUI support:** Includes both headless batch pipelines and an interactive registration GUI.
- **Supports multiple feature models:** Plug-and-play with various global and local feature extractors.
- **Optimized for scalability:** Designed to handle large buildings and multiple floor levels.

---

## 🚀 Main Mapping Pipeline

**Script:** `main_mapping_pipeline.py`

The recommended end-to-end mapping entrypoint. Runs the entire mapping pipeline from dense SLAM to 3D triangulation.

#### **Pipeline Stages**

1. **Dense Visual SLAM**  
2. **Equirectangular-to-perspective slicing**  
3. **Feature extraction (local & global)**  
4. **Feature matching + geometric verification**  
5. **3D triangulation with COLMAP (known poses)**  

#### **Command-line Usage**

```sh
python main_mapping_pipeline.py <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>
```

#### **Python API Example**

```python
from config import UNavConfig
from mapper.slam_runner import run_stella_vslam_dense
from mapper.slicer import slice_perspectives
from mapper.feature_extractor import extract_features_from_dir
from mapper.matcher import generate_and_stream_colmap
from mapper.colmap_triangulator import run_colmap_triangulation

# Step 1: Initialize configuration
config = UNavConfig(
    data_temp_root="/mnt/data/UNav-IO/temp",
    data_final_root="/mnt/data/UNav-IO/data",
    mapping_place="New_York_City",
    mapping_building="LightHouse",
    mapping_floor="4_floor",
    global_descriptor_model="DinoV2Salad"
)
mapper_config = config.mapping_config

# Step 2: Run the full pipeline
run_stella_vslam_dense(mapper_config)               # (1) Dense SLAM
slice_perspectives(mapper_config)                   # (2) Perspective slicing
extract_features_from_dir(mapper_config)            # (3) Feature extraction
generate_and_stream_colmap(mapper_config)           # (4) Feature matching + verification
run_colmap_triangulation(mapper_config)             # (5) 3D triangulation
```

---

## 🗺️ Floorplan-SLAM Alignment GUI

**Script:** `aligner.py`

An interactive tool to align the reconstructed 3D SLAM map with a 2D architectural floorplan image. This step is **critical** for enabling metric localization and reliable path planning.

#### **Python API Example**

```python
from config import UNavConfig
from mapper.aligner import run_aligner_gui

# Setup config (identical to mapping pipeline)
config = UNavConfig(
    data_temp_root="/mnt/data/UNav-IO/temp",
    data_final_root="/mnt/data/UNav-IO/data",
    mapping_place="New_York_City",
    mapping_building="LightHouse",
    mapping_floor="4_floor"
)
mapper_config = config.mapping_config

run_aligner_gui(mapper_config)
```

**Recommended:**  
- Complete all mapping steps before running the aligner.  
- The output transformation matrix will be saved and used in downstream localization and navigation modules.

---

## 🛰️ Visual Inspection: Mapping Quality

**Notebook:** `./unav/visualize_mapping.ipynb`

After completing the pipeline, use this Jupyter notebook to visually inspect the mapping results, validate alignment, and assess reconstruction quality.  
Recommended for **every new building/floor mapping session.**

---

## 🛠️ Configuration Example

**Location:** `config.py`

```python
from config import UNavConfig

config = UNavConfig(
    data_temp_root="/mnt/data/UNav-IO/temp",
    data_final_root="/mnt/data/UNav-IO/data",
    mapping_place="New_York_City",
    mapping_building="LightHouse",
    mapping_floor="4_floor",
    global_descriptor_model="DinoV2Salad"
)
```

---

## 📁 Directory Structure

```
mapper/
│
├── main_mapping_pipeline.py
├── aligner.py
├── slam_runner.py
├── slicer.py
├── feature_extractor.py
├── matcher.py
├── colmap_triangulator.py
├── tools/
│   ├── aligner/
│   ├── matcher/
│   └── slam/
├── config.py
└── README.md
```

---

## 💡 Tips & Best Practices

- **Absolute paths:** Always use absolute paths for all input/output directories.
- **Intermediate outputs:** Results are grouped by `<place>/<building>/<floor>`.
- **Out-of-memory:** If you hit GPU OOM (out-of-memory), adjust batch sizes in `matcher.py`.
- **Alignment:** The floorplan alignment GUI is the final mapping step before localization.
- **Extensibility:** Add new feature models by extending the relevant extractor classes.

---

## 👤 Maintainer

- **Developer:** Anbang Yang (`ay1620@nyu.edu`)
- **Last updated:** 2025-05-27

---

**For bug reports or feature requests, please contact the maintainer.**

