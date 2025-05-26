# UNav Mapping Pipeline

This repository provides the complete pipeline for constructing mapping datasets for the UNav system.  
It includes SLAM-based map generation, perspective slicing, feature extraction, matching, triangulation, registration, and dataset preparation for downstream localization and navigation tasks.

---

## 📍 Step 1: Generate SLAM Map

To generate the initial SLAM map using **stella_vslam_dense**, please follow the detailed instructions in  
➡ [SLAM/stella_vslam/README.md](SLAM/stella_vslam/README.md)

**Outputs produced:**
- Camera trajectory (`trajectory.txt`)
- Dense point cloud (`output_cloud.ply`)
- Keyframes (`keyframes/`)
- Serialized map database (`final_map.msg`)

These outputs are essential for all subsequent steps.

---

## 📍 Step 2: Slice Perspective Images

From the SLAM keyframes (`keyframes/`), generate perspective images covering the 360° FOV.

**Tools:**
```
unav_mapping/mapper/slicer.py
```

**Outputs:**
- Perspective images (`perspectives/`)
- Perspective poses (`perspective_poses.txt`)

---

## 📍 Step 3: Feature Extraction

Extract both **local features (e.g., SuperPoint + LightGlue)** and **global descriptors (e.g., MixVPR, DinoV2Salad)** from the sliced perspectives.

**Tools:**
```
unav_mapping/mapper/feature_extractor.py
```

**Outputs:**
- Global features (`global_features_{model_name}.h5`)
- Local features (`local_features.h5`)

---

## 📍 Step 4: Image Matching & Pairs Generation

(To be added)

---

## 📍 Step 5: COLMAP Triangulation Input Preparation

Prepare COLMAP-compatible input files:
- `images.txt`
- `cameras.txt`
- (optional) `pairs.txt`

> Use SLAM keyframe poses and camera parameters.

---

## 📍 Step 6: Registration & Segmentation (Optional)

Align the sparse SLAM point cloud with the floorplan.
- Estimate transformation matrix.
- Segment the map into rooms for efficient localization.

---

## 📍 Step 7: Convert to UNav Localization Format

Convert COLMAP outputs and other prepared data into the UNav-compatible localization database format.

**Outputs:**
- UNav localization database (`unav_localization_db/`)

---

## ⚙ Directory Structure Recommendation

```
/mnt/data/UNav-IO/temp/
├── <PLACE>/<BUILDING>/<FLOOR>/
│   ├── stella_vslam_dense/       # SLAM outputs
│   ├── perspectives/             # Sliced perspectives
│   ├── features/                 # Extracted features
│   ├── pairs.txt
│   ├── colmap_db/                # COLMAP workspace
│   ├── slam_to_floorplan.npy     # Optional registration matrix
│   └── room_segments/            # Optional segmented room maps
└── ...
```

---

## ✅ Notes

- Ensure all paths, parameters, and model configurations are consistent across modules.
- All configs are managed via `unav_mapping/config.py` (`UNavMappingConfig`).
- It is recommended to carefully validate outputs at each step using provided test notebooks and scripts.

---

## 📄 Modules

| Module                              | Purpose                             |
|--------------------------------------|-------------------------------------|
| `SLAM/stella_vslam/`                 | Run dense SLAM                      |
| `unav_mapping/mapper/slicer.py`      | Slice perspective images            |
| `unav_mapping/mapper/feature_extractor.py` | Extract local & global features    |
| (To be added)                        | Image matching & pairs generation  |
| (To be added)                        | COLMAP preparation & triangulation  |
| (To be added)                        | UNav localization file conversion   |

---
