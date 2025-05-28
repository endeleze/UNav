import math
from typing import List, Dict, Any, Tuple, Literal
from shapely.geometry import LineString, Point

def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-180, 180] degrees.

    Args:
        angle: Angle in degrees.

    Returns:
        Normalized angle in [-180, 180].
    """
    return (angle + 180) % 360 - 180

def generate_commands(
    path_coords: List[Tuple[float, float]],
    initial_heading: float,
    scale: float
) -> List[str]:
    """
    Generate clock-face-style navigation commands for a path.

    Args:
        path_coords: List of (x, y) coordinates for the path.
        initial_heading: Initial heading in degrees.
        scale: Meters-per-pixel scaling factor.

    Returns:
        List of command strings (turns and forward movements).
    """
    commands: List[str] = []
    heading = initial_heading

    for (x0, y0), (x1, y1) in zip(path_coords, path_coords[1:]):
        dx = x1 - x0
        dy = y0 - y1  # y inverted for screen coordinates
        bearing = math.degrees(math.atan2(dy, dx))

        turn = normalize_angle(bearing - heading)
        raw = -turn
        clock_num = int(round(raw / 30)) % 12
        hour = 12 if clock_num == 0 else clock_num

        # Turn instructions
        if hour == 12:
            commands.append("Proceed 12 o'clock")
        elif hour == 6:
            commands.append("Make a U-turn (6 o'clock)")
        else:
            mag = abs(turn)
            if mag < 30:
                qualifier = "Slight"
            elif mag < 60:
                qualifier = "Turn"
            else:
                qualifier = "Sharp"
            direction = "left" if turn > 0 else "right"
            commands.append(f"{qualifier} {direction} to {hour} o'clock")

        # Forward movement
        dist_m = math.hypot(dx, dy) * scale
        commands.append(f"Forward {dist_m:.2f} m")

        heading = bearing

    return commands

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
    Generate natural-language navigation commands from a full path result.

    Args:
        navigator: FacilityNavigator instance.
        path_result: Result dict from the path planner.
        initial_heading: Starting heading in degrees.
        unit: "meter" or "feet".

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
    
    # 1. Starting room announcement
    if len(keys) > 1 and keys[1] != "VIRT":
        fk0 = keys[1].rsplit("__", 1)[0]
        pf0 = navigator.pf_map[fk0]
        commands.append(f"You are currently in {pf0.get_current_room(coords[0])} of {fk0}.")
    else:
        commands.append("Starting navigation.")

    last_bearing = initial_heading
    heading = initial_heading

    # 2. Step-by-step navigation instructions
    for i in range(len(coords) - 1):
        key0, key1 = keys[i], keys[i + 1]
        desc1 = descriptions[i + 1].lower()

        # --- Parse keys into semantic fields ---
        if key0.count("__") == 3 and key1.count("__") == 3:
            place0, building0, floor0, nodeid0 = key0.split("__")
            place1, building1, floor1, nodeid1 = key1.split("__")

            # --- A. Detect place transition ---
            if place0 != place1:
                commands.append(f"You are approaching the transition to {place1}.")
                commands.append(f"Proceed to {floor1} of {building1} in {place1}.")
                heading = initial_heading  # Optionally reset or update heading
                continue  # Skip regular turn/forward for this segment

            # --- B. Detect building transition ---
            if building0 != building1:
                commands.append(f"You are approaching the transition to building {building1}.")
                commands.append(f"Proceed to {floor1} of {building1}.")
                heading = initial_heading  # Optionally reset or update heading
                continue  # Skip regular turn/forward for this segment

            # --- C. Detect floor transition ---
            if floor0 != floor1:
                # Insert a pre-transition announcement for the transition mode
                if "staircase" in desc1:
                    commands.append("You are approaching the staircase.")
                elif "elevator" in desc1:
                    commands.append("You are approaching the elevator.")
                elif "escalator" in desc1:
                    commands.append("You are approaching the escalator.")
                else:
                    commands.append("You are approaching the floor transition area.")

                # Generate floor transition instruction
                try:
                    # Try to extract numeric floor identifiers
                    floor_num_0 = int(''.join(filter(str.isdigit, floor0)))
                    floor_num_1 = int(''.join(filter(str.isdigit, floor1)))
                    direction = "up" if floor_num_1 > floor_num_0 else "down"
                except Exception:
                    # Fallback if floor parsing fails
                    direction = "up" if floor1 > floor0 else "down"

                if "staircase" in desc1:
                    commands.append(f"Go {direction} to {floor1} of {building1} via the staircase.")
                elif "elevator" in desc1:
                    commands.append(f"Press the {direction} button to {floor1} of {building1} using the elevator.")
                elif "escalator" in desc1:
                    commands.append(f"Take the escalator {direction} to {floor1} of {building1}.")
                else:
                    commands.append(f"Proceed to {floor1} of {building1}.")

                # Update heading after floor transition (if applicable)
                # Assumes you have a mapping to obtain connection points for heading adjustment
                if hasattr(navigator, 'pf_map'):
                    try:
                        nid = int(nodeid1) if nodeid1.isdigit() else 0
                        floor_pf = navigator.pf_map[f"{building1}__{floor1}"]
                        start_pt, end_pt = floor_pf.partner_lines[nid]
                        dxl = end_pt[0] - start_pt[0]
                        dyl = start_pt[1] - end_pt[1]  # y-inversion for screen coordinates
                        incident_bearing = math.degrees(math.atan2(dyl, dxl))
                        heading = incident_bearing
                    except Exception:
                        pass  # If data not available, skip heading update

                continue  # Skip regular turn/forward for this segment

        # -- B. Regular turning instructions --
        floor_key = key1.rsplit("__", 1)[0]
        scale = navigator.scales[floor_key]

        p0, p1 = coords[i], coords[i + 1]
        dx, dy = p1[0] - p0[0], p0[1] - p1[1]
        bearing = math.degrees(math.atan2(dy, dx))
        turn = normalize_angle(bearing - heading)
        raw = -turn
        clock_n = int(round(raw / 30)) % 12
        hour = 12 if clock_n == 0 else clock_n

        if hour == 12:
            commands.append("Proceed 12 o'clock")
        elif hour == 6:
            commands.append("Make a U-turn (6 o'clock)")
        else:
            mag = abs(turn)
            qual = "Slight" if mag < 30 else "Turn" if mag < 60 else "Sharp"
            direction = "left" if turn > 0 else "right"
            commands.append(f"{qual} {direction} to {hour} o'clock")

        # -- C. Forward distance --
        dist_m = math.hypot(dx, dy) * scale
        last_bearing = bearing

        # -- D. Door crossing detection --
        if key0 != "VIRT":
            fk = key0.rsplit("__", 1)[0]
            pf = navigator.pf_map[fk]
            line = LineString([p0, p1])
            for door_poly, _ in pf.door_polygons:
                if line.crosses(door_poly):
                    proj_px = line.project(door_poly.centroid)
                    distance_str = convert_distance(proj_px * scale, unit)
                    commands.append(f"Go through a door in {distance_str}")
                    break

        # -- E. Forward movement --
        distance_str = convert_distance(dist_m, unit)
        commands.append(f"Forward {distance_str}")
        heading = bearing

    # 3. Arrival announcement
    final_label = labels[-1]
    desc = descriptions[-1]

    if desc == 'center':
        commands.append(f"You will arrive {final_label}")
    else:
        desc_to_bearing = {
            'up': 90.0,
            'right': 0.0,
            'down': -90.0,
            'left': 180.0
        }
        orientation_bearing = desc_to_bearing.get(desc, last_bearing)
        turn = normalize_angle(orientation_bearing - last_bearing)
        raw = -turn
        clock_n = int(round(raw / 30)) % 12
        hour = 12 if clock_n == 0 else clock_n
        commands.append(f"{final_label} on {hour} o'clock")

    return commands

def split_path_by_floor(
    path_keys: List[str],
    path_coords: List[Tuple[float, float]]
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Split a global path into floor-specific segments.

    Args:
        path_keys: List of node keys (including "VIRT").
        path_coords: List of coordinates.

    Returns:
        Dictionary mapping "Building__Floor" to coordinates on that floor.
    """
    floor_segs: Dict[str, List[Tuple[float, float]]] = {}
    start_coord = None
    start_inserted = False

    for key, coord in zip(path_keys, path_coords):
        if key == "VIRT":
            start_coord = coord
            continue

        floor_key, _ = key.rsplit("__", 1)

        if floor_key not in floor_segs:
            floor_segs[floor_key] = []
            if start_coord is not None and not start_inserted:
                floor_segs[floor_key].append(start_coord)
                start_inserted = True

        floor_segs[floor_key].append(coord)

    return floor_segs
