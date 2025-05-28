import os
import json
import math
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional

from unav.navigator.pathfinder import PathFinder
from unav.navigator.snap import snap_inside_walkable
from unav.config import UNavNavigationConfig

class FacilityNavigator:
    """
    A multi-place, multi-building, multi-floor unified navigation system supporting inter-floor and inter-building routing
    with explicit support for labeled inter-waypoints (e.g., stairs, elevators).
    """

    def __init__(self, config: UNavNavigationConfig):
        """
        Initialize the navigation system with a unified configuration.

        Args:
            config (UNavNavigationConfig): Navigation configuration.
        """
        self.config = config

        # Map keys are in the form: "place__building__floor"
        self.pf_map: Dict[str, PathFinder] = {}
        self.inter_waypoints: Dict[str, List[str]] = {}  # label → list of full node keys (with place)
        for place, buildings in self.config.building_jsons.items():
            for bld, floors in buildings.items():
                for fl, json_path in floors.items():
                    key = f"{place}__{bld}__{fl}"
                    pf = PathFinder(json_path)
                    self.pf_map[key] = pf

        self.scales = self._load_scales(self.config.scale_file)
        self.G = nx.DiGraph()
        self._build_unified_graph()

    def _load_scales(self, scale_file: Optional[str]) -> Dict[str, float]:
        """
        Load scaling factors for each floor from a JSON file. Used for metric conversion.
        Keys are in the form: "place__building__floor".

        Args:
            scale_file (str): Path to JSON file containing scale factors.

        Returns:
            Dict[str, float]: Mapping from floor key to scale value.
        """
        scales = {key: 1.0 for key in self.pf_map}
        if scale_file and os.path.exists(scale_file):
            with open(scale_file) as f:
                data = json.load(f)
            for place, buildings in data.items():
                for bld, floors in buildings.items():
                    for fl, sc in floors.items():
                        key = f"{place}__{bld}__{fl}"
                        if key in scales:
                            scales[key] = sc
        return scales

    def _build_unified_graph(self):
        """
        Construct a unified directed graph over all places, buildings, and floors.
        This graph contains all walkable connections and explicitly connects inter-waypoints (e.g., stairs, elevators).
        """
        # Add all intra-floor edges.
        for key, pf in self.pf_map.items():
            scale = self.scales.get(key, 1.0)
            for u, v, d in pf.G.edges(data=True):
                uid = f"{key}__{u}"
                vid = f"{key}__{v}"
                scaled_weight = d['weight'] * scale
                self.G.add_edge(uid, vid, weight=scaled_weight)

            # Collect all inter-waypoints (group 4) by label; node_key includes place.
            for nid in pf.nav_ids:
                if pf.group_ids.get(nid) == 4:
                    label = pf.labels[nid]
                    if not label:
                        continue
                    node_key = f"{key}__{nid}"
                    self.inter_waypoints.setdefault(label, []).append(node_key)

        # Add edges between all inter-waypoints of the same label (across floors/buildings).
        for label, nodes in self.inter_waypoints.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    u, v = nodes[i], nodes[j]

                    # Parse floor and description for jump penalty.
                    bld_fl_u, nid_u = u.rsplit("__", 1)
                    pf_u = self.pf_map[bld_fl_u]
                    desc = pf_u.descriptions.get(int(nid_u), "").lower()
                    try:
                        _, _, floor_u = bld_fl_u.split("__")
                        _, _, floor_v = v.split("__")[:3]
                        floor_u_num = int(''.join(filter(str.isdigit, floor_u)))
                        floor_v_num = int(''.join(filter(str.isdigit, floor_v)))
                        jump = abs(floor_u_num - floor_v_num)
                    except Exception:
                        jump = 0

                    # Penalty for traversing inter-waypoints, different for stairs/elevators.
                    if "staircase" in desc:
                        penalty_per_jump = 50.0
                        total_penalty = penalty_per_jump * jump
                    elif "elevator" in desc:
                        total_penalty = 3.0
                    else:
                        total_penalty = 0.0

                    self.G.add_edge(u, v, weight=total_penalty)
                    self.G.add_edge(v, u, weight=total_penalty)

    def list_destinations(self) -> Dict[Tuple[str, str, str, int], Tuple[str, Tuple[float, float]]]:
        """
        List all available destinations across all places, buildings, and floors.

        Returns:
            Dict[Tuple[str, str, str, int], Tuple[str, Tuple[float, float]]]:
                Keys are (place, building, floor, dest_id).
                Values are (label, coordinates).
        """
        out = {}
        for key, pf in self.pf_map.items():
            place, bld, fl = key.split("__")
            for did in pf.dest_ids:
                out[(place, bld, fl, did)] = (pf.labels[did], pf.nodes[did])
        return out

    def find_path(
        self,
        start_place: str,
        start_building: str,
        start_floor: str,
        start_xy: Tuple[float, float],
        dest_place: str,
        dest_building: str,
        dest_floor: str,
        dest_id: int
    ) -> Dict[str, Any]:
        """
        Compute the shortest valid path from a given coordinate to a specified destination ID,
        possibly crossing places, buildings, or floors.

        Args:
            start_place (str): Name of the starting place.
            start_building (str): Name of the starting building.
            start_floor (str): Name of the starting floor.
            start_xy (tuple): (x, y) coordinates of the starting point.
            dest_place (str): Name of the destination place.
            dest_building (str): Name of the destination building.
            dest_floor (str): Name of the destination floor.
            dest_id (int): Destination node ID.

        Returns:
            Dict: {
                "path_coords": List[Tuple[float, float]],    # Path coordinates in metric/floorplan units.
                "path_labels": List[str],                    # Labels for each path node.
                "path_keys": List[str],                      # Full unique keys for each path node.
                "path_descriptions": List[str],              # Descriptions (stairs, elevator, etc).
                "total_cost": float,                         # Total path cost (sum of edge weights).
                "error": Optional[str]                       # Error message, if pathfinding failed.
            }
        """
        start_key = f"{start_place}__{start_building}__{start_floor}"
        target_key = f"{dest_place}__{dest_building}__{dest_floor}__{dest_id}"
        pf0 = self.pf_map[start_key]

        # Snap starting location inside walkable area if outside.
        start_xy = snap_inside_walkable(start_xy, pf0.walkable_union)

        # Create a temporary virtual node representing the user's real start coordinate.
        virt = "VIRT"
        self.G.add_node(virt)
        for nid in pf0.nav_ids + pf0.dest_ids:
            if pf0._visible(start_xy, pf0.nodes[nid]):
                w = math.hypot(start_xy[0] - pf0.nodes[nid][0], start_xy[1] - pf0.nodes[nid][1])
                self.G.add_edge(virt, f"{start_key}__{nid}", weight=w)

        try:
            path = nx.dijkstra_path(self.G, virt, target_key)
        except nx.NetworkXNoPath:
            self.G.remove_node(virt)
            return {"error": "No path found"}

        # Assemble path information.
        coords = []
        labels = []
        keys = []
        descriptions = []
        cost = 0.0
        prev_pt = start_xy

        for node in path:
            keys.append(node)
            if node == virt:
                coords.append(start_xy)
                labels.append("(start)")
                descriptions.append("")
                continue

            node_split = node.split("__")
            if len(node_split) < 4:
                continue
            place, bld, fl, nid_str = node_split
            floor_key = f"{place}__{bld}__{fl}"
            pf = self.pf_map[floor_key]
            nid = int(nid_str)
            pt = pf.nodes[nid]
            coords.append(pt)
            labels.append(pf.labels[nid])

            # Add human-friendly descriptions for navigation cues.
            if pf.group_ids[nid] == 4:
                desc = pf.descriptions.get(nid, "")
            elif pf.group_ids[nid] == 5:
                desc = pf.dest_orientations.get(nid, "")
            else:
                desc = ""
            descriptions.append(desc)

            cost += math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
            prev_pt = pt

        self.G.remove_node(virt)

        return {
            "path_coords": coords,
            "path_labels": labels,
            "path_keys": keys,
            "path_descriptions": descriptions,
            "total_cost": cost
        }
