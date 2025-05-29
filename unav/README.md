# 🗺️ Floorplan Labeling Guide for Multi-Floor Navigation

This guide describes the labeling standards used to annotate indoor environments for visual and accessible navigation, supporting waypoint planning, door detection, inter-floor traversal, and destination guidance.

---

## 1. 📐 Region Polygons (`group_id` 0–2)

| `group_id` | Type         | Label Example    | Description                              |
|------------|--------------|------------------|------------------------------------------|
| `0`        | Walkable     | `Room 101`       | Navigable area; optionally name the room |
| `1`        | Obstacle     | *(empty)*        | Blocks navigation; subtracts from walkable region |
| `2`        | Door         | `door`           | Helps detect door-crossing events        |

- Use `polygon` or `rectangle` shapes.
- Obstacles must **enclose** their areas.
- Doors can be small rectangles or narrow polygons at entrances.

---

## 2. 📍 Waypoints & Destinations (`group_id` 3–5)

| `group_id` | Type            | Label Example     | Description (`description` field)            | Notes                                        |
|------------|-----------------|-------------------|----------------------------------------------|---------------------------------------------|
| `3`        | Waypoint         | `w1` *(optional)* | *(empty or unique)*                          | Regular intra-floor navigation node         |
| `4`        | Inter-Waypoint   | `e1`, `s2`, `etc.` | `"elevator"` , `"staircase"` , `etc.` *(required)*    | Connects same-label nodes across floors     |
| `5`        | Destination      | `restroom`        | `'up'`, `'right'`, `'left'`, `'down'`, `'center'` *(required)*| Indicates the **spatial orientation** of the destination |

- All annotations are of `point` type.
- Each inter-waypoint (group 4) **must appear on all connected floors** with the **same label**.
- Each destination (group 5) must have a `description` value from:  
  `'up'`, `'right'`, `'down'`, `'left'`, `'center'`.

---

## 3. ➤ Inter-Waypoint Entry Direction (`group_id` 6)

To specify the **entry direction** into an inter-waypoint (for smoother command generation), use short lines:

| `group_id` | Shape | Label Example | Description                    |
|------------|--------|----------------|--------------------------------|
| `6`        | `line` | `e1`, `s2`, …   | Matches the inter-waypoint label. Direction = from first point to second point. |

- Must be a `line` with **exactly 2 points**.
- Placed in the direction **into** the group 4 node.
- Required for each group 4 inter-waypoint.

---

## 4. 🛗 Example: Multi-Floor Elevator `"e3"`

To connect an elevator `"e3"` across floors 3, 4, and 6:

1. On each floor:
    - Add a `point`:
      - `group_id = 4`
      - `label = "e3"`
      - `description = "elevator"`

    - Add a `line`:
      - `group_id = 6`
      - `label = "e3"`
      - From hallway → elevator node

2. System will connect all `"e3"` nodes virtually for multi-floor traversal, and use the partner line to orient the turn command upon arrival.

---

## 5. 🧾 Summary of Annotation Schema

| Group | Shape     | Required | Fields Used                     | Purpose                        |
|-------|-----------|----------|----------------------------------|--------------------------------|
| `0`   | polygon   | ✅        | `label` *(optional)*             | Walkable area                  |
| `1`   | polygon   | ✅        | *(none)*                         | Obstacle                       |
| `2`   | polygon   | ⚠️ Optional | `label` *(optional)*         | Door detection                 |
| `3`   | point     | ✅        | `label` *(optional)*             | Intra-floor waypoint           |
| `4`   | point     | ✅        | `label`, `description` *(req.)*  | Inter-floor connection         |
| `5`   | point     | ✅        | `label`, `description` *(req.)*  | Destination with orientation   |
| `6`   | line      | ✅ if 4   | `label` = group 4 label          | Entry direction into inter-waypoint |

---

## 6. ✅ Validation Checklist

- [x] All `group_id = 4` inter-waypoints **appear on every connected floor**
- [x] Every group 4 point has a matching **entry line (group 6)** with the same label
- [x] All `group_id = 5` destinations include a valid `description`: `'up'`, `'down'`, `'left'`, `'right'`, `'center'`
- [x] Doors (group 2) are clearly labeled and distinguishable from walls
- [x] Obstacles (group 1) fully **enclose** their regions

---

## 7. 📌 Tips for LabelMe Usage

- Use consistent zoom level and spacing when placing points.
- Snap group 6 lines cleanly into inter-waypoint dots.
- Don’t reuse labels across unrelated points (e.g., two `"e1"`s on the same floor with different meanings).
- Keep lines short and horizontal/vertical if possible for clarity.

---

For issues or updates, please contact the project maintainer.