import json
import math
import networkx as nx
from shapely.geometry import Polygon as ShapelyPolygon, Point, LineString
from shapely.ops import unary_union
from typing import Dict, List, Tuple, Any

class PathFinder:
    """
    Constructs a directed visibility graph over manually defined waypoints (groups 3, 4),
    enabling shortest path queries from any position to final destinations (group 5).
    
    Takes into account:
        - Walkable areas (group 0, with optional room labels)
        - Obstacles (group 1)
        - Doors (group 2)
    """

    def __init__(self, json_path: str):
        # Node metadata
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.labels: Dict[int, str] = {}
        self.group_ids: Dict[int, int] = {}
        self.descriptions: Dict[int, str] = {}  # For inter-waypoints
        self.dest_orientations: Dict[int, str] = {}  # For destination nodes
        self.nav_ids: List[int] = []
        self.dest_ids: List[int] = []
        self.partner_lines: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

        # Region geometries
        self.walkable_polygons: List[ShapelyPolygon] = []
        self.obstacle_polygons: List[ShapelyPolygon] = []
        self.door_polygons: List[Tuple[ShapelyPolygon, str]] = []
        self.room_polygons: List[Tuple[ShapelyPolygon, str]] = []

        # Unified walkable region after merging and subtracting
        self.walkable_union: ShapelyPolygon = None

        # Visibility graph
        self.G = nx.DiGraph()

        # Build graph from data
        self._load_data(json_path)
        self._build_graph()

    def _euclidean(self, p1, p2) -> float:
        """Return Euclidean distance between two points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _load_data(self, json_path: str):
        """Load map data from the given JSON file and parse regions and nodes."""
        with open(json_path) as f:
            data = json.load(f)

        node_id = 0
        raw_group6: List[Dict[str, Any]] = []

        # 1) First pass: load regions (groups 0–2) and points (groups 3–5)
        for shape in data["shapes"]:
            stype = shape.get("shape_type")
            gid = shape.get("group_id")
            pts = shape.get("points", [])
            label = (shape.get("label") or "").strip()
            desc = (shape.get("description") or "").strip()

            # Polygon and rectangle regions (walkable, obstacles, doors)
            if stype in ("polygon", "rectangle"):
                if stype == "rectangle" and len(pts) == 2:
                    (x0, y0), (x1, y1) = pts
                    pts = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
                poly = ShapelyPolygon(pts)

                if gid == 0:
                    self.walkable_polygons.append(poly)
                    self.room_polygons.append((poly, label))
                elif gid == 1:
                    self.obstacle_polygons.append(poly)
                elif gid == 2:
                    self.door_polygons.append((poly, label))
                continue

            # Points: navigation points, inter-waypoints, destinations
            if stype == "point" and pts:
                pt = tuple(pts[0])
                self.nodes[node_id] = pt
                self.labels[node_id] = label
                self.group_ids[node_id] = gid
                if gid == 4:
                    self.descriptions[node_id] = desc
                if gid in (3, 4):
                    self.nav_ids.append(node_id)
                if gid == 5:
                    self.dest_ids.append(node_id)
                    self.dest_orientations[node_id] = desc
                node_id += 1

            # Lines (group 6): companion lines for inter-waypoints
            if stype == "line" and gid == 6 and len(pts) == 2:
                raw_group6.append({
                    "label": label,
                    "points": pts
                })

        # 2) Associate group 6 lines with inter-waypoints by label
        for entry in raw_group6:
            pts = entry["points"]
            line = (tuple(pts[0]), tuple(pts[1]))
            lbl = entry["label"]
            for nid, node_lbl in self.labels.items():
                if self.group_ids.get(nid) == 4 and node_lbl == lbl:
                    self.partner_lines[nid] = line
                    break

        # 3) Merge walkable + door regions and subtract obstacles
        merged = unary_union(self.walkable_polygons + [poly for poly, _ in self.door_polygons])
        for obs in self.obstacle_polygons:
            merged = merged.difference(obs)
        self.walkable_union = merged

    def _visible(self, p1, p2) -> bool:
        """
        Return True if the line segment p1→p2 stays within the walkable area
        and does not intersect any obstacle.
        """
        line = LineString([p1, p2])
        if not self.walkable_union.contains(line):
            return False
        for obs in self.obstacle_polygons:
            if line.crosses(obs) or line.within(obs):
                return False
        return True

    def _build_graph(self):
        """Construct the directed visibility graph."""
        # Connect navigational nodes
        for i in self.nav_ids:
            for j in self.nav_ids:
                if i < j and self._visible(self.nodes[i], self.nodes[j]):
                    w = self._euclidean(self.nodes[i], self.nodes[j])
                    self.G.add_edge(i, j, weight=w)
                    self.G.add_edge(j, i, weight=w)

        # Connect navigational nodes to destinations
        for nid in self.nav_ids:
            for did in self.dest_ids:
                if self._visible(self.nodes[nid], self.nodes[did]):
                    w = self._euclidean(self.nodes[nid], self.nodes[did])
                    self.G.add_edge(nid, did, weight=w)

    def find_path(self, start_id: int, dest_id: int) -> Dict[str, Any]:
        """
        Run Dijkstra's algorithm between two node IDs.

        Args:
            start_id: ID of the start node.
            dest_id: ID of the destination node.

        Returns:
            A dictionary with path coordinates, labels, IDs, and total cost.
        """
        if dest_id not in self.dest_ids:
            return {"error": "Destination must be terminal"}
        if start_id == dest_id:
            return {
                "path_ids": [start_id],
                "path_coords": [self.nodes[start_id]],
                "path_labels": [self.labels[start_id]],
                "total_cost": 0.0
            }
        try:
            path = nx.dijkstra_path(self.G, source=start_id, target=dest_id)
            coords = [self.nodes[n] for n in path]
            cost = sum(self._euclidean(coords[i], coords[i + 1]) for i in range(len(coords) - 1))
            return {
                "path_ids": path,
                "path_coords": coords,
                "path_labels": [self.labels[n] for n in path],
                "total_cost": cost
            }
        except nx.NetworkXNoPath:
            return {"error": "No path found"}

    def find_path_from_pose(
        self,
        pose_xy: Tuple[float, float],
        dest_id: int
    ) -> Dict[str, Any]:
        """
        Insert a virtual node at the user’s pose, connect it to all visible nodes,
        and compute the shortest path to a destination.

        Args:
            pose_xy: (x, y) coordinate of the user's pose.
            dest_id: ID of the destination.

        Returns:
            A path dictionary similar to `find_path()`.
        """
        if dest_id not in self.dest_ids:
            return {"error": "Destination must be terminal"}

        # Add virtual node for pose
        vid = -1
        self.nodes[vid] = pose_xy
        self.labels[vid] = "pose"
        self.group_ids[vid] = -1
        self.G.add_node(vid)

        # Connect to visible navigation nodes
        for nid in self.nav_ids:
            if self._visible(pose_xy, self.nodes[nid]):
                w = self._euclidean(pose_xy, self.nodes[nid])
                self.G.add_edge(vid, nid, weight=w)
        
        # Connect directly to visible destinations (if any)
        for did in self.dest_ids:
            if self._visible(pose_xy, self.nodes[did]):
                w = self._euclidean(pose_xy, self.nodes[did])
                self.G.add_edge(vid, did, weight=w)

        # Run shortest path algorithm
        raw = self.find_path(vid, dest_id)

        # Remove virtual node
        if self.G.has_node(vid):
            self.G.remove_node(vid)
        self.nodes.pop(vid, None)
        self.labels.pop(vid, None)
        self.group_ids.pop(vid, None)

        # Prepend pose to result path if found
        if "path_coords" in raw and raw["path_coords"]:
            raw["path_coords"].insert(0, pose_xy)
            raw["path_labels"].insert(0, "start_pose")
            raw["path_ids"].insert(0, vid)

        return raw

    def get_current_room(self, pose_xy: Tuple[float, float]) -> str:
        """
        Return the label of the room polygon that contains the given point.
        
        Args:
            pose_xy: (x, y) coordinate to check.

        Returns:
            The room label, or "Unknown" if not found.
        """
        pt = Point(*pose_xy)
        for poly, lbl in self.room_polygons:
            if poly.contains(pt):
                return lbl or "Unnamed Room"
        return "Unknown"

    def list_all_destinations(self) -> Dict[int, Tuple[str, Tuple[float, float]]]:
        """List all known destination nodes with their labels and coordinates."""
        return {d: (self.labels[d], self.nodes[d]) for d in self.dest_ids}

    def get_destination_id_by_name(self, name: str) -> int:
        """Find a destination ID by case-insensitive substring match on its label."""
        for d in self.dest_ids:
            if name.lower() in self.labels[d].lower():
                return d
        return None
