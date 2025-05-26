import os
import json
import math
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional

from navigator.pathfinder import PathFinder
from navigator.snap import snap_inside_walkable

from config import UNavNavigationConfig

class FacilityNavigator:
    """
    A multi-building, multi-floor pathfinding system using named inter-waypoints (group 4).
    """

    def __init__(
        self,
        config: UNavNavigationConfig
    ):
        """
        Initialize FacilityNavigator using unified navigation configuration.

        Args:
            config (UNavNavigationConfig): Navigation configuration object.
        """
        self.config = config
        
        self.pf_map: Dict[str, PathFinder] = {}
        self.inter_waypoints: Dict[str, List[str]] = {}  # label → list of node_keys

        for place, buildings in self.config.building_jsons.items():
            for bld, floors in buildings.items():
                for fl, json_path in floors.items():
                    key = f"{bld}__{fl}"
                    pf = PathFinder(json_path)
                    self.pf_map[key] = pf

        self.scales = self._load_scales(self.config.scale_file)
        self.G = nx.DiGraph()
        self._build_unified_graph()

    def _load_scales(self, scale_file: Optional[str]) -> Dict[str, float]:
        """Load scale values for each floor from a JSON file."""
        scales = {key: 1.0 for key in self.pf_map}
        if scale_file and os.path.exists(scale_file):
            with open(scale_file) as f:
                data = json.load(f)
            for place in data.values():
                for bld, floors in place.items():
                    for fl, sc in floors.items():
                        key = f"{bld}__{fl}"
                        if key in scales:
                            scales[key] = sc
        return scales

    def _build_unified_graph(self):
        """Construct a unified directed graph across all buildings and floors."""

        # Add intra-floor edges
        for key, pf in self.pf_map.items():
            scale = self.scales.get(key, 1.0)
            for u, v, d in pf.G.edges(data=True):
                uid = f"{key}__{u}"
                vid = f"{key}__{v}"
                scaled_weight = d['weight'] * scale
                self.G.add_edge(uid, vid, weight=scaled_weight)

            # Collect inter-waypoints (group 4) by label
            for nid in pf.nav_ids:
                if pf.group_ids.get(nid) == 4:
                    label = pf.labels[nid]
                    if not label:
                        continue
                    node_key = f"{key}__{nid}"
                    self.inter_waypoints.setdefault(label, []).append(node_key)

        # Connect all inter-waypoints with the same label across floors
        for label, nodes in self.inter_waypoints.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    u, v = nodes[i], nodes[j]

                    # Determine connection type using the "description" field
                    bld_fl_u, nid_u = u.rsplit("__", 1)
                    pf_u = self.pf_map[bld_fl_u]
                    desc = pf_u.descriptions.get(int(nid_u), "").lower()
                    floor_u = int(bld_fl_u.split("__")[1][0])  # e.g. '6_floor' -> 6
                    floor_v = int(v.split("__")[1][0])
                    jump = abs(floor_u - floor_v)
                    # Assign penalty weight based on type and floor difference
                    if "staircase" in desc:
                        penalty_per_jump = 50.0
                        total_penalty = penalty_per_jump * jump
                    elif "elevator" in desc:
                        total_penalty = 3.0
                    else:
                        total_penalty = 0.0

                    self.G.add_edge(u, v, weight=total_penalty)
                    self.G.add_edge(v, u, weight=total_penalty)

    def list_destinations(self) -> Dict[Tuple[str, str, int], Tuple[str, Tuple[float, float]]]:
        """List all known destinations across all buildings and floors."""
        out = {}
        for key, pf in self.pf_map.items():
            bld, fl = key.split("__", 1)
            for did in pf.dest_ids:
                out[(bld, fl, did)] = (pf.labels[did], pf.nodes[did])
        return out

    def find_path(
        self,
        start_building: str,
        start_floor: str,
        start_xy: Tuple[float, float],
        dest_building: str,
        dest_floor: str,
        dest_id: int
    ) -> Dict[str, Any]:
        """
        Find the shortest path from a start coordinate to a destination ID.

        Args:
            start_building: Starting building name.
            start_floor: Starting floor name.
            start_xy: (x, y) coordinate of the starting point.
            dest_building: Destination building name.
            dest_floor: Destination floor name.
            dest_id: Destination node ID.

        Returns:
            A dictionary containing path coordinates, labels, keys, descriptions, and total cost.
        """
        start_key = f"{start_building}__{start_floor}"
        target_key = f"{dest_building}__{dest_floor}__{dest_id}"
        pf0 = self.pf_map[start_key]

        # Snap slightly off-point to nearest walkable area
        start_xy = snap_inside_walkable(start_xy, pf0.walkable_union)

        # Add a virtual starting node to connect from the user's real location
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

        # Prepare output data
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
                descriptions.append("")  # No description for virtual start
                continue

            floor_key, nid_str = node.rsplit("__", 1)
            pf = self.pf_map[floor_key]
            nid = int(nid_str)

            pt = pf.nodes[nid]
            coords.append(pt)
            labels.append(pf.labels[nid])

            # Get descriptions based on group type
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
