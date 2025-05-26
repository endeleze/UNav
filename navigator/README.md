# 🧭 UNav Navigation System

A multi-floor, multi-building indoor navigation module that supports visual pathfinding, spoken-style command generation, and segment-based path planning. Designed for applications in visually impaired assistance, robotics, and digital twin simulations.

## ✨ Features

- ✅ Multi-floor and multi-building visual pathfinding
- ✅ Inter-waypoints for stairs/elevator transitions
- ✅ Spoken-style navigation commands (e.g., “Turn left to 10 o'clock”)
- ✅ Snapping of noisy or invalid poses to the nearest walkable space
- ✅ Modular structure for easy extension and integration

## 🛠 Installation

Make sure you have Python 3.7+ and `pip` installed. Install it:

```bash
pip install git+https://github.com/ai4ce/UNav_Navigation.git
```

This installs the `unav` module for import.

## 🚀 Quick Start

```python
from unav_navigator import FacilityNavigator, commands_from_result

# Load navigation system
building_jsons = {
    "LightHouse": {
        "3_floor": "example_data/New_York_City/LightHouse/3_floor/boundaries.json",
        "4_floor": "example_data/New_York_City/LightHouse/4_floor/boundaries.json",
        "6_floor": "example_data/New_York_City/LightHouse/6_floor/boundaries.json",
    }
}
nav = FacilityNavigator(building_jsons, scale_file=scale_file)

# Sample a valid start pose
floor_key, (x, y), theta = 'LightHouse__4_floor', (1560.0018488325186, 667.4292020409354), 2.1883365724167443

# Choose a destination
dest_building, dest_floor, selected_index = "LightHouse", "3_floor", 3
target_key = f"{dest_building}__{dest_floor}"

# Plan path
pf_target = nav.pf_map[target_key]
dest_id = pf_target.dest_ids[selected_index]

result = nav.find_path("LightHouse", "4_floor", (x, y), dest_building, dest_floor, dest_id)

# Generate navigation instructions
commands = commands_from_result(nav, result, initial_heading=theta)
for i, cmd in enumerate(commands, 1):
    print(f"{i}. {cmd}")
```

## 📦 Data Annotation Format

To learn how to label floorplans and define waypoints, doors, and walkable space, please refer to:

📄 [`example_data/README.md`](example_data/README.md)

## 📄 License

MIT License © 2025 Anbang Yang