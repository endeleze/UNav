import os
import math
import random
from typing import Dict, Tuple, List, Optional

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from shapely.geometry import Point
import networkx as nx

from navigator.pathfinder import PathFinder
from navigator.commander import split_path_by_floor

# =================== Random Sampling Utilities ===================

def sample_random_pose(
    pf_map: Dict[str, PathFinder]
) -> Tuple[str, Tuple[float, float], float]:
    """
    Randomly sample a valid pose (x, y, theta) on any floor (including building).

    Args:
        pf_map (Dict[str, PathFinder]): Mapping from "building__floor" to PathFinder instance.

    Returns:
        Tuple: (floor_key, (x, y), theta)
    """
    key = random.choice(list(pf_map.keys()))
    pf = pf_map[key]
    region = pf.walkable_union
    minx, miny, maxx, maxy = region.bounds

    while True:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        pt = Point(x, y)
        if not region.contains(pt):
            continue
        if any(obs.contains(pt) for obs in pf.obstacle_polygons):
            continue
        theta = random.uniform(0, 2 * math.pi)
        return key, (x, y), theta

def perturb_pose_off_walkable_pose(
    pf_map: Dict[str, PathFinder],
    max_offset: float = 150.0,
    max_attempts: int = 100
) -> Tuple[str, Tuple[float, float], float]:
    """
    Sample a pose slightly outside the walkable region to simulate noisy localization.

    Args:
        pf_map (Dict[str, PathFinder]): Floor mapping.
        max_offset (float): Maximum perturbation distance in pixels.
        max_attempts (int): Number of tries before aborting.

    Returns:
        Tuple: (floor_key, (x, y), theta) where (x, y) is outside the walkable area.
    """
    keys = list(pf_map.keys())
    random.shuffle(keys)
    for key in keys:
        pf = pf_map[key]
        walkable = pf.walkable_union
        minx, miny, maxx, maxy = walkable.bounds
        for _ in range(max_attempts):
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            pt = Point(x, y)
            if not walkable.contains(pt):
                continue
            # Push the point outward in a random direction
            angle = random.uniform(0, 2 * math.pi)
            dx = math.cos(angle) * random.uniform(10, max_offset)
            dy = math.sin(angle) * random.uniform(10, max_offset)
            pt2 = Point(x + dx, y + dy)
            if not walkable.contains(pt2):
                theta = random.uniform(0, 2 * math.pi)
                return key, (pt2.x, pt2.y), theta
    raise RuntimeError("Failed to generate a perturbed pose outside walkable space.")

# =================== Floorplan and Navigation Plotting ===================

def plot_floorplan(
    ax,
    walkable_polys: List,
    obstacle_polys: List,
    door_polys: Optional[List] = None
) -> None:
    """
    Render walkable and obstacle regions using Matplotlib patches.

    Args:
        ax: Matplotlib axis.
        walkable_polys (List): List of walkable shapely polygons.
        obstacle_polys (List): List of obstacle shapely polygons.
        door_polys (Optional[List]): List of door polygons.
    """
    walk_patches = [MplPolygon(list(poly.exterior.coords), closed=True) for poly in walkable_polys]
    obs_patches = [MplPolygon(list(poly.exterior.coords), closed=True) for poly in obstacle_polys]
    ax.set_aspect("equal")
    ax.set_title("Floorplan Visualization")
    ax.add_collection(PatchCollection(walk_patches, facecolor='green', alpha=0.3, edgecolor='none'))
    ax.add_collection(PatchCollection(obs_patches,  facecolor='gray',  alpha=0.5, edgecolor='none'))
    if door_polys:
        door_patches = [MplPolygon(list(poly.exterior.coords), closed=True) for poly in door_polys]
        ax.add_collection(PatchCollection(door_patches, facecolor='saddlebrown', alpha=0.6, edgecolor='none'))

def plot_points(
    ax,
    waypoints: List[Tuple[Tuple[float, float], int]],
    inter_waypoints: List[Tuple[Tuple[float, float], str, int]],
    destinations: List[Tuple[Tuple[float, float], str]],
    selected_dest: Optional[Tuple[Tuple[float, float], str]] = None
) -> None:
    """
    Plot navigation points, inter-waypoints, and destinations on the floorplan.

    Args:
        ax: Matplotlib axis.
        waypoints: List of (point, id) tuples.
        inter_waypoints: List of (point, label, id) tuples.
        destinations: List of (point, name) tuples.
        selected_dest: Optional selected destination.
    """
    for pt, _ in waypoints:
        ax.add_patch(Circle(pt, radius=15, color='blue', alpha=0.8))
    for pt, label, _ in inter_waypoints:
        ax.add_patch(Circle(pt, radius=15, color='orange', alpha=0.9))
        ax.text(pt[0] + 20, pt[1], label, fontsize=7, color='darkorange', va='center')
    for pt, name in destinations:
        if selected_dest and pt == selected_dest[0]:
            ax.plot(pt[0], pt[1], marker='*', markersize=20, color='red')
            ax.text(pt[0] + 20, pt[1], name, fontsize=9, color='darkred', va='center')
        else:
            ax.add_patch(Circle(pt, radius=15, color='red', alpha=0.9))
            ax.text(pt[0] + 20, pt[1], name, fontsize=8, color='darkred', va='center')

def plot_pose(ax, x: float, y: float, theta: float) -> None:
    """
    Plot a pose as a point with heading arrow and angle annotation.

    Args:
        ax: Matplotlib axis.
        x, y: Position coordinates.
        theta: Heading in radians.
    """
    dx = 100 * math.cos(theta)
    dy = 100 * math.sin(theta)
    ax.plot(x, y, marker='o', markersize=10, color='black')
    ax.arrow(x, y, dx, dy, head_width=50, head_length=50, fc='black', ec='black', linewidth=2)
    ax.text(x + 60, y + 60, f"θ={math.degrees(theta):.1f}°", fontsize=8, color='black')

def draw_graph_on_floorplans(
    G: nx.DiGraph,
    pf_map: Dict[str, PathFinder],
    data_root: str,
    place: str,
    floor_keys: List[str],
    show_virt: bool = False,
    figsize: Tuple[int, int] = (8, 14)
) -> None:
    """
    Draw navigation graphs over floorplans, including node group coloring and edge weights.

    Args:
        G (nx.DiGraph): Navigation graph.
        pf_map (Dict[str, PathFinder]): PathFinder objects per floor.
        place (str): Root directory for floorplan backgrounds.
        floor_keys (List[str]): Floors to visualize (e.g., ["Building__3_floor"]).
        show_virt (bool): Whether to draw VIRT node edges.
        figsize (Tuple[int, int]): Figure size per subplot.
    """
    n = len(floor_keys)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0]*n, figsize[1]))
    if n == 1:
        axes = [axes]

    for ax, floor_key in zip(axes, floor_keys):
        bld, flr = floor_key.split("__", 1)
        pf = pf_map[floor_key]

        # Load floorplan background image if available
        bg_path = os.path.join(data_root, place, bld, flr, "floorplan.png")
        if os.path.exists(bg_path):
            bg = mpimg.imread(bg_path)
            ax.imshow(bg, extent=[0, bg.shape[1], bg.shape[0], 0])
        else:
            print(f"[WARNING] Background not found for {floor_key}")

        node_pos = {}
        color_map = {}
        sizes = {}
        labels = {}

        for nid, pt in pf.nodes.items():
            full_key = f"{floor_key}__{nid}"
            node_pos[full_key] = pt
            labels[full_key] = pf.labels.get(nid, "")
            gid = pf.group_ids.get(nid, -1)
            if gid == 4:
                color_map[full_key] = "orange"
                sizes[full_key] = 300
            elif gid == 5:
                color_map[full_key] = "red"
                sizes[full_key] = 200
            else:
                color_map[full_key] = "blue"
                sizes[full_key] = 100

        # Optionally, include VIRT node for special navigation scenarios
        if show_virt:
            for u, v, d in G.edges(data=True):
                if u == "VIRT" and v.startswith(floor_key):
                    node_pos[v] = node_pos.get(v)
                    color_map[v] = "green"
                    sizes[v] = 120

        sub_nodes = list(node_pos.keys())
        subG = G.subgraph(sub_nodes)

        # Draw navigation edges
        nx.draw_networkx_edges(
            subG, pos=node_pos, ax=ax,
            arrows=True, edge_color='limegreen', width=1.5
        )

        # Draw edge weights as green text at the middle of edges
        edge_labels = {
            (u, v): f"{round(d['weight'])}" for u, v, d in subG.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            subG,
            pos=node_pos,
            edge_labels=edge_labels,
            font_size=6,
            font_color='green',
            ax=ax,
            rotate=False,
            label_pos=0.5
        )

        # Draw node groups (by color)
        for color in set(color_map.values()):
            group_nodes = [n for n in subG.nodes if color_map[n] == color]
            nx.draw_networkx_nodes(
                subG, pos=node_pos, nodelist=group_nodes,
                node_color=color, node_size=[sizes[n] for n in group_nodes], ax=ax,
                edgecolors='black' if color == "orange" else 'none'
            )

        nx.draw_networkx_labels(subG, pos=node_pos, labels=labels, font_size=7, ax=ax)
        ax.set_title(f"{bld}/{flr}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_navigation_path(
    nav,
    result: dict,
    start_pose: Tuple[float, float, float],
    data_root: str,
    start_building: str,
    start_floor: str,
    dest_building: str,
    dest_floor: str,
    selected_pt: Tuple[float, float],
    selected_name: str,
    place: str
) -> None:
    """
    Plot the computed navigation path over the corresponding floorplan backgrounds.

    Args:
        nav: Navigation object, must contain .pf_map, .scales.
        result (dict): Navigation result containing "path_coords" and "path_keys".
        start_pose (Tuple[float, float, float]): (x, y, theta) of start.
        start_building (str): Building of start.
        start_floor (str): Floor of start.
        dest_building (str): Building of destination.
        dest_floor (str): Floor of destination.
        selected_pt (Tuple[float, float]): Coordinates of selected destination.
        selected_name (str): Name of the selected destination.
        place (str): Dataset root for background images.
    """
    if 'error' in result:
        return

    x, y, theta = start_pose
    full_coords = result["path_coords"]
    floor_segs = split_path_by_floor(result["path_keys"], full_coords)
    floors = list(floor_segs.keys())

    # Find used inter-waypoints for multi-floor transitions
    red_inter_pts = set()
    path_keys = result["path_keys"]
    for i in range(len(path_keys) - 1):
        k0, k1 = path_keys[i], path_keys[i + 1]
        if k0.count("__") == 2 and k1.count("__") == 2:
            b0, f0, nid_str = k0.split("__")
            b1, f1, _ = k1.split("__")
            if (b0, f0) != (b1, f1):
                pf = nav.pf_map[f"{b0}__{f0}"]
                nid = int(nid_str)
                if pf.group_ids.get(nid) == 4:
                    red_inter_pts.add(pf.nodes[nid])

    fig, axes = plt.subplots(1, len(floors), figsize=(8 * len(floors), 14))
    if len(floors) == 1:
        axes = [axes]

    for ax, floor_key in zip(axes, floors):
        bld, flr = floor_key.split("__", 1)
        bg = mpimg.imread(os.path.join(data_root, place, bld, flr, "floorplan.png"))
        ax.imshow(bg, extent=[0, bg.shape[1], bg.shape[0], 0])

        pf = nav.pf_map[floor_key]
        w, o, d = pf.walkable_polygons, pf.obstacle_polygons, [poly for poly, _ in pf.door_polygons]
        plot_floorplan(ax, w, o, d)

        segment = floor_segs[floor_key]
        used_xy_set = set(segment)

        # Highlight used inter-waypoints
        for i in pf.nav_ids:
            if pf.group_ids[i] == 4:
                pt = pf.nodes[i]
                if pt in used_xy_set:
                    color = 'red' if pt in red_inter_pts else 'orange'
                    ax.plot(pt[0], pt[1], 'o', markersize=14, markeredgewidth=2,
                            markeredgecolor=color, markerfacecolor='none')

        # Mark the selected destination
        if floor_key == f"{dest_building}__{dest_floor}":
            ax.plot(*selected_pt, marker='*', markersize=16, color='red')
            ax.text(*selected_pt, selected_name, fontsize=12, color='red')

        # Plot the start pose
        if floor_key == f"{start_building}__{start_floor}":
            plot_pose(ax, x, y, theta)

        # Draw path
        xs, ys = zip(*segment)
        ax.plot(xs, ys, color='lime', linewidth=4)

        # Path length annotation
        length_m = sum(
            math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            for i in range(1, len(xs))
        ) * nav.scales.get(floor_key, 1.0)
        print(f"[INFO] Floor {flr} segment length: {length_m:.2f} m")

        ax.set_title(f"{bld}/{flr}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
