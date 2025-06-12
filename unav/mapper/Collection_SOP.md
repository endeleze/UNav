# 360 Camera Mapping Video Collection

## Overview

This document describes the Standard Operating Procedure (SOP) for collecting 360-degree mapping videos for indoor environments using any 360 camera. The aim is to ensure high-quality, comprehensive data for SLAM, detailed mapping, and robust Visual Place Recognition (VPR). This guide also includes steps for establishing floorplan scale via real-world measurements.

---

## 1. Device Preparation

- Use a 360-degree camera capable of high-resolution video recording (preferably 5K or higher).
- Ensure batteries are fully charged and sufficient storage is available.
- Set the correct date and time on the device.
- Clean all camera lenses before recording.

---

## 2. Pre-collection Preparation

- Assign at least one assistant to help open doors during the collection process.
- Prepare a printed or digital floorplan for path planning and scale measurement.
- Ensure all target areas are accessible and well-lit.
- Remove unnecessary obstacles from the path.

---

## 3. Coverage and Loop Strategy

Each floorplan must be covered with **at least three loops**, ensuring all areas (corridors, rooms, corners) are included:

### Loop 1: Main Path (SLAM Loop Closure)
- Walk along main corridors and primary pathways, forming a closed loop if possible.
- Assistants pre-open all doors along the route.
- Maintain steady walking speed and smooth movement.
- Camera orientation must always face the walking direction.

### Loop 2: Small Rooms and Side Areas (Detail Mapping)
- Enter all small rooms and side areas along the main path.
- Assistants pre-open doors before entering each space.
- Make a loop inside each room, then return to the main path.
- Keep the camera facing forward in the walking direction.

### Loop 3: Door Interaction (VPR Diversity)
- Ensure all doors are closed prior to this loop.
- The collector personally opens each door before entering and closes it after exiting.
- Walk the same or similar path as previous loops, interacting with every door.
- Maintain the camera’s forward-facing orientation.

**Note:** Strive to cover every accessible area of the floorplan. Make additional passes if necessary to ensure complete coverage.

---

## 4. Video Output Requirements

- Export videos at the highest feasible resolution, preferably 5K.
- During export or post-processing, fix the camera view direction to always align with the walking direction.
- Avoid sudden camera rotations or erratic movements.
- Name exported video files using the convention:  
  `[Site]-[Floor]-[Loop]-[Date]-[Operator].mp4`

---

## 5. Floorplan Scale Measurement Procedure

1. **Pre-marking:**  
   - On the printed/digital floorplan, select and mark several (at least 3, ideally more) point pairs that are easily identifiable in the real environment.

2. **On-site Measurement:**  
   - At the actual location, use a tape measure or laser distance meter to accurately measure the real-world distance between each marked point pair.

3. **Recording Measurements:**  
   - Document each pair’s floorplan coordinates (pixels or drawing units) and the corresponding real-world measurement (in meters or centimeters).

4. **Calculating Scale:**  
   - Calculate the scale factor for each pair as:  
     `Scale = Real-world distance / Floorplan distance`
   - Compute the average scale across all pairs for increased accuracy.

5. **Documentation:**  
   - Include the raw data and the final average scale in the session log.

---

## 6. Documentation and Data Handling

- For each session, log: location, floor, date, camera model, operator(s), assistant(s), start/end times, and any anomalies.
- Mark loop sequence and area coverage on the floorplan and store with video data.
- Immediately back up all raw footage to at least two separate locations (e.g., external SSD and cloud storage).
- Conduct a quick quality check for completeness and clarity.

---

## 7. Additional Recommendations

- Synchronize 360 camera data with additional sensors (IMU, GPS) if available.
- Use stabilization accessories or software as needed.
- Notify site occupants of the collection schedule to avoid interference.

---
