# Map Labeling & Multi-language

This page covers:

1. How to annotate `boundaries.json` with `labelme`
2. How to run UNav translation GUI for multilingual labels

Reference examples:

- `https://github.com/ai4ce/UNav_Navigation/tree/main/example_data`

## Part A: Map Labeling with Labelme

### Output File

Save annotation file as:

```text
<DATA_FINAL_ROOT>/<place>/<building>/<floor>/boundaries.json
```

### Group and Shape Rules (Code-Aligned)

The parser in `unav/navigator/pathfinder.py` expects the following:

| group_id | shape_type | meaning | required fields |
|---|---|---|---|
| 0 | polygon or rectangle | walkable room/region | `label` optional room name |
| 1 | polygon or rectangle | obstacle (non-walkable) | - |
| 2 | polygon or rectangle | door area (added to walkable union) | `label` optional door name |
| 3 | point | normal navigation waypoint | recommended `label=waypoint` |
| 4 | point | inter-floor waypoint (stairs/elevator connector) | `label` connector ID, `description` should be `staircase` or `elevator` |
| 5 | point | destination | `label` destination name, `description` orientation hint (`left/right/center/up/...`) |
| 6 | line | companion line for group 4 waypoint | `label` must match group 4 label |
| 7 | line | corridor **skeleton / "railway"** trunk — a straight centreline segment; endpoints coincide with other group-7 segments at junctions | `label` optional; use `shape_type=line` (2-point, straight) |

### Labelme Authoring Procedure

1. Open floorplan image in `labelme`.
2. Draw walkable polygons (`group_id=0`).
3. Draw obstacles (`group_id=1`).
4. Draw doors (`group_id=2`).
5. Place navigation waypoints (`group_id=3`).
6. Place stair/elevator transfer points (`group_id=4`).
7. Place destination points (`group_id=5`).
8. Draw companion lines for each group 4 connector (`group_id=6`, same label).
9. Lay the corridor **skeleton / "railway"** (`group_id=7`) — see [Corridor Skeleton Network](#corridor-skeleton-railway-network-group_id7).
10. Save JSON as `boundaries.json` in floor folder.

### Corridor Skeleton (Railway) Network (`group_id=7`)

Routing uses a **two-tier network**, like rail + road:

- **Railway (`group_id=7`)** — the long-haul corridor trunk lines. Straight centreline
  segments running down the middle of corridors. Route planning strongly prefers the
  railway for the long part of a trip, so corridors are followed as clean **straight legs
  with 90° turns**, instead of cutting a diagonal short-cut across a junction.
- **Roads (`group_id=3` waypoints)** — the local network. Used to get **on/off** the
  railway and for the **last mile** to a destination, and for **open spaces / outdoor**
  areas where straight-line travel is correct.

Think of a skeleton waypoint as the **first/last station of a straight rail line**: you
mark the *ends* of the track, not every stop in between.

**How to lay the railway**

1. Draw each straight corridor run as **one straight line** (`group_id=7`,
   `shape_type=line`) down the **centre** of the corridor.
2. Put a vertex **only at track ends, junctions, and turns**. **Do NOT** add a vertex at
   every door or intermediate point — a straight corridor is a **single line from end to
   end**.
3. Where corridors meet, make the line endpoints **coincide** (identical coordinates).
   That shared point is a **junction** — the train can switch tracks there.
4. At a corner / T-junction, use **two straight lines meeting at the corner vertex** —
   never one diagonal line across the open junction.
5. Lay railway **only inside corridors**. **Open spaces and outdoor areas get no railway**
   — they are handled by roads (`group_id=3` waypoints + line-of-sight).

**Keep it sparse.** The railway is a *skeleton*: a handful of straight lines per floor,
not a dense mesh. Rooms and destinations are reached **from** the railway by roads, so you
do **not** extend the railway to every door.

Why this matters: the railway makes the corridor geometry an explicit, hand-controlled
trunk, so there is no reliance on automatic corridor detection and no diagonal
corner-cutting. Destinations are then approached down the corridor centre and announced as
"on your left / right", rather than via a long hypotenuse straight to the door.

### Inter-floor Connector Rule

To connect floors, `group_id=4` labels must match across floors/buildings.

Example:

- `LH-e1` on floor 3 and `LH-e1` on floor 4 means same elevator shaft
- `description=elevator` or `description=staircase` controls cross-floor penalty logic

### Minimal JSON Example

```json
{
  "shapes": [
    {"label": "corridor", "group_id": 0, "shape_type": "polygon", "points": [[0,0],[100,0],[100,20],[0,20]]},
    {"label": "pillar", "group_id": 1, "shape_type": "polygon", "points": [[40,5],[50,5],[50,15],[40,15]]},
    {"label": "door_A", "group_id": 2, "shape_type": "rectangle", "points": [[98,8],[104,12]]},
    {"label": "waypoint", "group_id": 3, "shape_type": "point", "points": [[20,10]]},
    {"label": "LH-e1", "group_id": 4, "shape_type": "point", "description": "elevator", "points": [[80,10]]},
    {"label": "Main Elevator", "group_id": 5, "shape_type": "point", "description": "up", "points": [[82,10]]},
    {"label": "LH-e1", "group_id": 6, "shape_type": "line", "points": [[80,6],[80,14]]},
    {"label": "rail", "group_id": 7, "shape_type": "line", "points": [[20,10],[80,10]]}
  ]
}
```

## Part B: Multi-language Labels with UNav GUI

UNav translation editor writes labels to:

```text
<DATA_FINAL_ROOT>/_i18n/labels.json
```

### Option 1: Use your wrapper in `/home/unav/Desktop/unav-run`

```bash
cd /home/unav/Desktop/unav-run
./run_translator.sh -r <DATA_FINAL_ROOT> -H 127.0.0.1 -p 5001
```

Then open:

```text
http://127.0.0.1:5001
```

### Option 2: Run module directly from this repo

```bash
python -m unav.mapper.tools.i18n_label_web \
  --data-final-root <DATA_FINAL_ROOT> \
  --use-nav \
  --host 127.0.0.1 \
  --port 5001
```

### `--use-nav` vs file mode

- `--use-nav`: derives Place/Building/Floor/Destination tree from navigation assets (`boundaries.json`)
- no `--use-nav`: falls back to scanning floor folders and optional `destinations.json`

Fallback `destinations.json` format:

```json
[
  {"id": "101", "name": "Reception"},
  {"id": "102", "name": "Elevator"}
]
```

### `labels.json` Structure

```json
{
  "places": {
    "New_York_City": {"en": "New York City", "zh-Hans": "纽约"}
  },
  "buildings": {
    "New_York_City/LightHouse": {"en": "LightHouse", "zh-Hans": "灯塔楼"}
  },
  "floors": {
    "New_York_City/LightHouse/4_floor": {"en": "4F", "zh-Hans": "四层"}
  },
  "destinations": {
    "New_York_City/LightHouse/4_floor/79": {"en": "Reception", "zh-Hans": "接待处"}
  },
  "aliases": {
    "zh-Hans": {
      "接待处": "New_York_City/LightHouse/4_floor/79"
    }
  }
}
```

## Final Validation Checklist

- `boundaries.json` exists and loads without parse errors
- All required `group_id` categories are present
- Corridor skeleton (`group_id=7`) laid along corridor centres; endpoints coincide at junctions; none in open spaces
- `group_id=4` labels match across floors where transitions should exist
- `_i18n/labels.json` created and populated
- Target language labels resolve correctly in app/localization flow
