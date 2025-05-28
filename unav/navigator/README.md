# 🧭 UNav Navigation System

A robust, multi-floor, multi-building **indoor navigation module** for visual pathfinding, spoken-style command generation, and modular path planning.  
Designed for applications in **visually impaired assistance, robotics, digital twins, and smart buildings**.

---

## 📦 Prerequisites

Before using the navigation system, make sure you have the following:

- **Python 3.8+**
- All dependencies from your project `requirements.txt`
- **Shapely** and **networkx** (for graph and geometry operations)
- **labelme** for floorplan/waypoint annotation:
    - Annotate your floorplans and boundaries with [labelme](https://github.com/wkentaro/labelme).
- Properly prepared floorplan annotation JSONs for all floors and buildings you wish to use.

Optionally:
- `visualize_navigation.ipynb` is provided for visual debugging and path simulation.

---

## 🚀 Quick Start

### 1. **Load Floorplan Data into FacilityNavigator**

```python
from config import UNavConfig
from navigator.navigator import FacilityNavigator

DATA_FINAL_ROOT = "/mnt/data/UNav-IO/data"
PLACES = {
    "New_York_City": {
        "LightHouse": ["3_floor", "4_floor", "6_floor"]
    }
}

config = UNavConfig(
    data_final_root=DATA_FINAL_ROOT,
    places=PLACES
)
nav = FacilityNavigator(config.navigator_config)
```

---

### 2. **Select Destination from a Target Floor**

```python
target_place = "New_York_City"
target_building = "LightHouse"
target_floor = "6_floor"
target_key = (target_place, target_building, target_floor)

# Select a destination index (e.g., the 4th destination)
selected_index = 3
pf_target = nav.pf_map[target_key]
dest_id = pf_target.dest_ids[selected_index]
```

---

### 3. **Get a Starting Pose**

You can use the output of the UNav localizer as the starting pose, or for demo/testing, sample randomly:

```python
from navigator.tools.pose_sampling import sample_random_pose

key, (x, y), theta = sample_random_pose(nav.pf_map)
start_place, start_building, start_floor = key
```

---

### 4. **Find and Visualize a Multi-floor Path**

```python
result = nav.find_path(
    start_place, start_building, start_floor, (x, y),
    target_place, target_building, target_floor, dest_id
)
```

---

### 5. **Generate Spoken Navigation Commands**

```python
from navigator.tools.commands import commands_from_result
import math

cmds = commands_from_result(
    nav,
    result,
    initial_heading=-math.degrees(theta),
    unit="feet"   # or "meter"
)
for i, c in enumerate(cmds, 1):
    print(f"{i}. {c}")
```

---

### 6. **Visual Simulation**

To simulate and debug navigation visually, use:

```
./unav/visualize_navigation.ipynb
```

---

## ✨ Features

- ✅ Multi-floor, multi-building shortest path routing
- ✅ Named inter-waypoints for stairs/elevator transitions
- ✅ Spoken-style navigation commands (“Slight left to 11 o’clock”, etc.)
- ✅ Robust pose snapping to walkable regions (handles noisy inputs)
- ✅ Modular, extensible codebase for research and production

---

## 🗺️ Data Annotation Format

- Floorplan JSONs should be produced using [labelme](https://github.com/wkentaro/labelme).
- Annotate walkable areas, obstacles, doors, navigation nodes, inter-waypoints, and destinations according to project convention.
- See `unav/core/LABELME_README.md` for detailed guidelines.

---

## 🛠 Modular APIs

Each navigation stage is available as a Python function/class:
- `FacilityNavigator`: Unified interface for multi-floor pathfinding
- `PathFinder`: Floor-level waypoint graph constructor and search
- `snap_inside_walkable`: Utility to sanitize/snap noisy poses
- `commands_from_result`: Spoken-style command generator for navigation
- ...and more under `navigator/` and `navigator/tools/`

---

## 📁 Directory Structure

```
navigator/
│
├── navigator.py
├── pathfinder.py
├── snap.py
├── tools/
│   ├── commands.py
│   ├── pose_sampling.py
│   └── ...
└── README.md
```

---

## 💡 Tips & Best Practices

- Always use **absolute paths** for annotation files.
- Start/end locations should be safely within walkable regions (use `snap_inside_walkable` if in doubt).
- Tune penalty values for stairs/elevator transitions in the config as needed for your building.
- All navigation results are returned as Python dictionaries for downstream integration.
- Spoken-style command templates can be customized in `commands_from_result`.

---

## 👤 Maintainer

- **Developer:** Anbang Yang (`ay1620@nyu.edu`)
- **Last updated:** 2025-05-27

---

**For bug reports or feature requests, please contact the maintainer.**
