import math
from typing import List, Dict, Any, Tuple, Literal, Union
from shapely.geometry import LineString

def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-180, 180] degrees.

    Args:
        angle: Angle in degrees.

    Returns:
        Normalized angle in [-180, 180].
    """
    return (angle + 180) % 360 - 180

def convert_distance(meters: float, unit: Literal["meter", "feet"] = "meter") -> str:
    """
    Convert a distance from meters to a string in meters or feet.

    Args:
        meters: Distance in meters.
        unit: "meter" or "feet".

    Returns:
        String representation of the distance.
    """
    if unit == "feet":
        feet = meters * 3.28084
        feet_rounded = int(round(feet))
        return f"{feet_rounded} feet"
    elif unit == "meter":
        if meters < 1:
            rounded = round(meters, 1)
        else:
            rounded = round(meters * 2) / 2  # Nearest 0.5
        if rounded.is_integer():
            rounded = int(rounded)
        return f"{rounded} meter{'s' if rounded != 1 else ''}"
    else:
        raise ValueError("Unit must be 'meter' or 'feet'.")

def commands_from_result(
    navigator,
    path_result: Dict[str, Any],
    initial_heading: float,
    unit: Literal["meter", "feet"] = "meter"
) -> List[str]:
    """
    Generate natural-language navigation commands from a planned path result,
    merging door instructions with forward movement and optimizing turn/straight logic.

    Args:
        navigator: FacilityNavigator instance (should have pf_map, scales, etc).
        path_result: Dict containing path information (coordinates, keys, labels, descriptions).
        initial_heading: Starting heading in degrees.
        unit: Output unit for distances ("meter" or "feet").

    Returns:
        List of navigation instruction strings.
    """
    if 'error' in path_result:
        raise ValueError(f"Cannot generate commands: {path_result['error']}")

    coords = path_result["path_coords"]
    labels = path_result["path_labels"]
    keys = path_result["path_keys"]
    descriptions = path_result["path_descriptions"]

    commands = []

    # Announce starting location
    if len(keys) > 1 and keys[1] != "VIRT":
        if isinstance(keys[1], tuple) and len(keys[1]) == 4:
            floor_key = keys[1][:3]
            place, building, floor = floor_key
            pf0 = navigator.pf_map[floor_key]
            room = pf0.get_current_room(coords[0]) if hasattr(pf0, 'get_current_room') else ""
            commands.append(f"You are currently in {room} on {floor} of {building}, {place}.")
        else:
            commands.append("Starting navigation.")
    else:
        commands.append("Starting navigation.")

    # --- Main loop for navigation steps ---
    heading = initial_heading
    i = 0
    straight_distance = 0.0
    door_events = []
    accumulated_segments = []
    segment_start_idx = 0

    while i < len(coords) - 1:
        key0, key1 = keys[i], keys[i + 1]
        p0, p1 = coords[i], coords[i + 1]
        desc1 = descriptions[i + 1].lower()
        label1 = labels[i + 1]

        # Compute heading and bearing
        dx, dy = p1[0] - p0[0], p0[1] - p1[1]
        segment_dist = math.hypot(dx, dy)

        # Default scale (meters per pixel) for the current floor
        if isinstance(key1, tuple) and len(key1) == 4:
            floor_key = key1[:3]
            scale = navigator.scales.get(floor_key, 1.0)
        else:
            scale = 1.0

        # --- Handle region/building/floor transitions ---
        if (isinstance(key0, tuple) and len(key0) == 4 and
            isinstance(key1, tuple) and len(key1) == 4):

            place0, building0, floor0, nodeid0 = key0
            place1, building1, floor1, nodeid1 = key1

            # Place transition
            if place0 != place1:
                if straight_distance > 0:
                    dist_str = convert_distance(straight_distance * scale, unit)
                    if door_events:
                        door_pos = min(door_events, key=lambda d: d["dist"])
                        door_dist = convert_distance(door_pos["dist"] * scale, unit)
                        commands.append(f"Forward {dist_str} and go through a door in {door_dist}")
                        door_events.clear()
                    else:
                        commands.append(f"Forward {dist_str}")
                    straight_distance = 0.0
                commands.append(f"You are approaching the transition to {place1}.")
                commands.append(f"Proceed to {floor1} of {building1} in {place1}.")
                heading = initial_heading
                i += 1
                continue

            # Building transition
            if building0 != building1:
                if straight_distance > 0:
                    dist_str = convert_distance(straight_distance * scale, unit)
                    if door_events:
                        door_pos = min(door_events, key=lambda d: d["dist"])
                        door_dist = convert_distance(door_pos["dist"] * scale, unit)
                        commands.append(f"Forward {dist_str} and go through a door in {door_dist}")
                        door_events.clear()
                    else:
                        commands.append(f"Forward {dist_str}")
                    straight_distance = 0.0
                commands.append(f"You are approaching the transition to building {building1}.")
                commands.append(f"Proceed to {floor1} of {building1}.")
                heading = initial_heading
                i += 1
                continue

            # Floor transition
            if floor0 != floor1:
                if straight_distance > 0:
                    dist_str = convert_distance(straight_distance * scale, unit)
                    if door_events:
                        door_pos = min(door_events, key=lambda d: d["dist"])
                        door_dist = convert_distance(door_pos["dist"] * scale, unit)
                        commands.append(f"Forward {dist_str} and go through a door in {door_dist}")
                        door_events.clear()
                    else:
                        commands.append(f"Forward {dist_str}")
                    straight_distance = 0.0

                # Announce approach to transition
                if "staircase" in desc1:
                    commands.append("You are approaching the staircase.")
                elif "elevator" in desc1:
                    commands.append("You are approaching the elevator.")
                elif "escalator" in desc1:
                    commands.append("You are approaching the escalator.")
                else:
                    commands.append("You are approaching the floor transition area.")

                # Floor transition instruction
                try:
                    floor_num_0 = int(''.join(filter(str.isdigit, floor0)))
                    floor_num_1 = int(''.join(filter(str.isdigit, floor1)))
                    direction = "up" if floor_num_1 > floor_num_0 else "down"
                except Exception:
                    direction = "up" if floor1 > floor0 else "down"

                if "staircase" in desc1:
                    commands.append(f"Go {direction} to {floor1} of {building1} via the staircase.")
                elif "elevator" in desc1:
                    commands.append(f"Press the {direction} button to {floor1} of {building1} using the elevator.")
                elif "escalator" in desc1:
                    commands.append(f"Take the escalator {direction} to {floor1} of {building1}.")
                else:
                    commands.append(f"Proceed to {floor1} of {building1}.")
                heading = initial_heading
                i += 1
                continue

        # --- Turn and straight handling ---
        # Compute turn (change in heading)
        bearing = math.degrees(math.atan2(dy, dx))
        turn = normalize_angle(bearing - heading)
        raw = -turn
        clock_n = int(round(raw / 30)) % 12
        hour = 12 if clock_n == 0 else clock_n

        # Check if this step requires a turn
        is_turn = abs(turn) >= 25

        if is_turn:
            # Flush previous straight segment if any
            if straight_distance > 0:
                dist_str = convert_distance(straight_distance * scale, unit)
                if door_events:
                    door_pos = min(door_events, key=lambda d: d["dist"])
                    door_dist = convert_distance(door_pos["dist"] * scale, unit)
                    commands.append(f"Forward {dist_str} and go through a door in {door_dist}")
                    door_events.clear()
                else:
                    commands.append(f"Forward {dist_str}")
                straight_distance = 0.0

            if hour != 12 and hour != 6:
                qual = "Slight" if abs(turn) < 45 else "Turn" if abs(turn) < 90 else "Sharp"
                direction = "left" if turn > 0 else "right"
                commands.append(f"{qual} {direction} to {hour} o'clock")
            elif hour == 6:
                commands.append("Make a U-turn (6 o'clock)")
            heading = bearing

        # Accumulate straight segment distance
        straight_distance += segment_dist

        # Door detection (between p0 and p1)
        if key0 != "VIRT" and hasattr(navigator, "pf_map"):
            floor_key0 = key0[:3]
            pf = navigator.pf_map.get(floor_key0, None)
            if pf and hasattr(pf, "door_polygons"):
                line = LineString([p0, p1])
                for door_poly, _ in getattr(pf, "door_polygons", []):
                    if line.crosses(door_poly):
                        proj_px = line.project(door_poly.centroid)
                        door_events.append({"dist": proj_px, "idx": i})
                        break

        # If this is the last step, flush any remaining straight segment
        is_last = (i == len(coords) - 2)
        next_turn = False
        if not is_last:
            # Preview next step for turn
            p2 = coords[i + 2]
            dx2, dy2 = p2[0] - p1[0], p1[1] - p2[1]
            bearing2 = math.degrees(math.atan2(dy2, dx2))
            next_turn = abs(normalize_angle(bearing2 - heading)) >= 25

        if is_last or next_turn:
            if straight_distance > 0:
                dist_str = convert_distance(straight_distance * scale, unit)
                if door_events:
                    door_pos = min(door_events, key=lambda d: d["dist"])
                    door_dist = convert_distance(door_pos["dist"] * scale, unit)
                    commands.append(f"Forward {dist_str} and go through a door in {door_dist}")
                    door_events.clear()
                else:
                    commands.append(f"Forward {dist_str}")
                straight_distance = 0.0

        i += 1

    # Final arrival instruction with direction
    final_label = labels[-1]
    desc = descriptions[-1].lower()
    desc_to_bearing = {
        'up': 90.0,
        'right': 0.0,
        'down': -90.0,
        'left': 180.0
    }
    orientation_bearing = desc_to_bearing.get(desc, heading)
    turn = normalize_angle(orientation_bearing - heading)
    raw = -turn
    clock_n = int(round(raw / 30)) % 12
    hour = 12 if clock_n == 0 else clock_n

    # Direction wording follows standard clock (right=3, left=9)
    if hour == 12:
        dir_word = "ahead"
    elif hour == 6:
        dir_word = "behind"
    elif hour in (1, 2, 3, 4, 5):
        dir_word = "right"
    elif hour in (7, 8, 9, 10, 11):
        dir_word = "left"

    commands.append(f"{final_label} on {hour} o'clock {dir_word}")

    return commands


def split_path_by_floor(
    path_keys: List[Union[str, Tuple[str, str, str, int]]],
    path_coords: List[Tuple[float, float]]
) -> Dict[Tuple[str, str, str], List[Tuple[float, float]]]:
    """
    Split a global path into floor-specific segments using unique (place, building, floor) tuple as key.

    Args:
        path_keys: List of node keys (may include "VIRT" as a string, others as (place, building, floor, node_id)).
        path_coords: List of coordinates.

    Returns:
        Dict mapping (place, building, floor) to list of coordinates on that floor.
    """
    floor_segs: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = {}
    start_coord = None
    start_inserted = False

    for key, coord in zip(path_keys, path_coords):
        if key == "VIRT":
            start_coord = coord
            continue

        # key should be tuple (place, building, floor, node_id)
        floor_key = key[:3]  # (place, building, floor)

        if floor_key not in floor_segs:
            floor_segs[floor_key] = []
            if start_coord is not None and not start_inserted:
                floor_segs[floor_key].append(start_coord)
                start_inserted = True

        floor_segs[floor_key].append(coord)

    return floor_segs