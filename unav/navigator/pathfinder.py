import json
import math
import networkx as nx
from shapely.geometry import Polygon as ShapelyPolygon, Point, LineString, MultiLineString
from shapely.ops import unary_union, nearest_points
from typing import Dict, List, Tuple, Any, Optional

class PathFinder:
    """
    PathFinder builds a navigation graph from annotated floorplan data and answers
    shortest-path queries.

    Two-tier "railway + road" routing:
      - Railway (group 7): hand-drawn corridor centreline segments. Its graph has nodes
        ONLY at junctions / turns / ends (sparse). Routing rides the railway along
        corridors, giving straight legs and right-angle turns with no diagonal
        corner-cutting.
      - Roads (group 3/4 waypoints + line-of-sight): the local network, for open/outdoor
        spaces. Road edges are cost-penalised so the long part of a trip prefers the rail.

    Any external point (destination, current pose, road waypoint) is joined to the
    railway by its **perpendicular foot** on the nearest rail segment (the "normal
    distance"), splitting that segment at the foot. No densified sampling nodes.

    If a map has no railway (group 7), behaviour is identical to the classic
    line-of-sight (visibility) graph — fully backward compatible.

    Groups:
      - 0 walkable, 1 obstacle, 2 door
      - 3 navigation waypoint, 4 inter-floor waypoint, 5 destination
      - 6 companion line (for group 4), 7 corridor skeleton / railway line
    """

    # --- two-tier routing parameters ---
    RAIL_COST = 1.0        # railway edge weight multiplier (cheap -> preferred)
    ROAD_COST = 3.0        # road / on-off-ramp edge weight multiplier (penalised)
    RAIL_MERGE = 20.0      # merge railway endpoints within this distance (junction snap, px)
    RAIL_WELD_PX = 28.0    # auto-weld a loose endpoint onto a nearby segment / crossing (px)
    RAIL_ID_BASE = 1_000_000  # railway node id space (avoids collision with waypoints)
    # heading-aware start: absorb a short perpendicular entry hop so the first
    # instruction is "go forward" (not a spurious right-angle) when the user is
    # already walking along the corridor.
    START_HOP_PX = 90.0    # entry hop shorter than this may be absorbed
    START_ALIGN_DEG = 40.0 # ...if heading aligns with the corridor within this angle
    START_TURN_DEG = 50.0  # ...and keeping the hop would inject a turn larger than this

    def __init__(self, json_path: str):
        # Node (waypoint/destination) data
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.labels: Dict[int, str] = {}
        self.group_ids: Dict[int, int] = {}
        self.descriptions: Dict[int, str] = {}
        self.dest_orientations: Dict[int, str] = {}
        self.nav_ids: List[int] = []
        self.dest_ids: List[int] = []
        self.partner_lines: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

        # Region geometry
        self.walkable_polygons: List[ShapelyPolygon] = []
        self.obstacle_polygons: List[ShapelyPolygon] = []
        self.door_polygons: List[Tuple[ShapelyPolygon, str]] = []
        self.room_polygons: List[Tuple[ShapelyPolygon, str]] = []
        self.walkable_union: ShapelyPolygon = None

        # Railway (group 7): raw straight segments + built junction/foot nodes
        self.rail_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.rail_node_ids: List[int] = []
        self._next_rail_id = self.RAIL_ID_BASE
        self._road_mult = 1.0

        self.G = nx.DiGraph()
        self.route_network = None

        self._load_data(json_path)
        self._build_graph()

    # ------------------------------------------------------------------ helpers
    def _euclidean(self, p1, p2) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _proj_seg(p, a, b):
        """Perpendicular foot of p on segment a-b (clamped), plus param t in [0,1]."""
        ax, ay = a; bx, by = b; px, py = p
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 <= 1e-12:
            return (ax, ay), 0.0
        t = ((px - ax) * dx + (py - ay) * dy) / L2
        t = max(0.0, min(1.0, t))
        return (ax + t * dx, ay + t * dy), t

    @staticmethod
    def _bearing(p0, p1) -> float:
        """Heading convention shared with commander (y axis points down in image)."""
        return math.degrees(math.atan2(p0[1] - p1[1], p1[0] - p0[0]))

    @staticmethod
    def _norm_angle(a) -> float:
        return (a + 180.0) % 360.0 - 180.0

    # -------------------------------------------------------------------- load
    def _load_data(self, json_path: str):
        with open(json_path) as f:
            data = json.load(f)

        node_id = 0
        raw_group6: List[Dict[str, Any]] = []

        for shape in data["shapes"]:
            stype = shape.get("shape_type")
            gid = shape.get("group_id")
            pts = shape.get("points", [])
            label = (shape.get("label") or "").strip()
            desc = (shape.get("description") or "").strip()

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

            if stype == "line" and gid == 6 and len(pts) == 2:
                raw_group6.append({"label": label, "points": pts})

            # Group 7: corridor skeleton / railway. A line/linestrip is split into
            # straight 2-point segments (consecutive vertices).
            if stype in ("line", "linestrip") and gid == 7 and len(pts) >= 2:
                for a, b in zip(pts[:-1], pts[1:]):
                    self.rail_lines.append((tuple(a), tuple(b)))

        for entry in raw_group6:
            pts = entry["points"]
            line = (tuple(pts[0]), tuple(pts[1]))
            lbl = entry["label"]
            for nid, node_lbl in self.labels.items():
                if self.group_ids.get(nid) == 4 and node_lbl == lbl:
                    self.partner_lines[nid] = line
                    break

        merged = unary_union(self.walkable_polygons + [poly for poly, _ in self.door_polygons])
        for obs in self.obstacle_polygons:
            merged = merged.difference(obs)
        self.walkable_union = merged

    def _visible(self, p1, p2) -> bool:
        line = LineString([p1, p2])
        if not self.walkable_union.contains(line):
            return False
        for obs in self.obstacle_polygons:
            if line.crosses(obs) or line.within(obs):
                return False
        return True

    # -------------------------------------------------------------- graph build
    def _build_graph(self):
        has_rail = bool(self.rail_lines)
        self._road_mult = self.ROAD_COST if has_rail else 1.0
        m = self._road_mult

        # Road layer: line-of-sight among nav/inter waypoints (penalised when rail exists)
        for i in self.nav_ids:
            for j in self.nav_ids:
                if i < j and self._visible(self.nodes[i], self.nodes[j]):
                    w = self._euclidean(self.nodes[i], self.nodes[j]) * m
                    self.G.add_edge(i, j, weight=w, kind="road")
                    self.G.add_edge(j, i, weight=w, kind="road")
        for nid in self.nav_ids:
            for did in self.dest_ids:
                if self._visible(self.nodes[nid], self.nodes[did]):
                    w = self._euclidean(self.nodes[nid], self.nodes[did]) * m
                    self.G.add_edge(nid, did, weight=w, kind="road")

        if has_rail:
            self._build_railway()

        seen = set()
        segments = []
        for u, v in self.G.edges():
            key = (min(u, v), max(u, v))
            if key not in seen and u in self.nodes and v in self.nodes:
                seen.add(key)
                segments.append((self.nodes[u], self.nodes[v]))
        self.route_network = MultiLineString(segments) if segments else None

    # ------------------------------------------------------------- railway core
    def _new_rail_node(self, p) -> int:
        nid = self._next_rail_id
        self._next_rail_id += 1
        self.nodes[nid] = (float(p[0]), float(p[1]))
        self.labels[nid] = "rail"
        self.group_ids[nid] = 7
        self.rail_node_ids.append(nid)
        self.G.add_node(nid)
        return nid

    def _junction_node(self, p) -> int:
        """Return an existing rail node within RAIL_MERGE of p, else create one."""
        for nid in self.rail_node_ids:
            if self._euclidean(p, self.nodes[nid]) < self.RAIL_MERGE:
                return nid
        return self._new_rail_node(p)

    def _add_rail_edge(self, u, v):
        w = self._euclidean(self.nodes[u], self.nodes[v]) * self.RAIL_COST
        self.G.add_edge(u, v, weight=w, kind="rail")
        self.G.add_edge(v, u, weight=w, kind="rail")

    def _remove_rail_edge(self, u, v):
        if self.G.has_edge(u, v):
            self.G.remove_edge(u, v)
        if self.G.has_edge(v, u):
            self.G.remove_edge(v, u)

    def _unique_rail_edges(self):
        seen = set()
        out = []
        for u, v, d in self.G.edges(data=True):
            if d.get("kind") == "rail":
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    out.append((u, v))
        return out

    @staticmethod
    def _seg_intersection(p1, p2, p3, p4):
        """Proper interior intersection of segments p1p2 and p3p4, else None."""
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-9:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
        if 1e-6 < t < 1 - 1e-6 and 1e-6 < u < 1 - 1e-6:
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return None

    def _weld_railway(self):
        """
        Make junctions robust to sloppy hand annotation: the annotator only has to draw
        lines that get *close* at a junction (within RAIL_WELD_PX), not pixel-perfect.

          (a) a loose endpoint landing on the interior of another track is welded onto it
              (T-junction), and
          (b) two tracks that cross without a shared vertex are split and welded at the
              crossing (cross-junction).
        """
        # (a) endpoint -> segment interior
        changed = True
        while changed:
            changed = False
            for n in list(self.rail_node_ids):
                pn = self.nodes[n]
                best = None
                for u, v in self._unique_rail_edges():
                    if n == u or n == v:
                        continue
                    foot, t = self._proj_seg(pn, self.nodes[u], self.nodes[v])
                    if t <= 1e-3 or t >= 1 - 1e-3:
                        continue  # foot at an endpoint -> covered by endpoint clustering
                    d = self._euclidean(pn, foot)
                    if d < self.RAIL_WELD_PX and (best is None or d < best[0]):
                        best = (d, u, v)
                if best is not None:
                    _, u, v = best
                    self._remove_rail_edge(u, v)   # weld the through-track through n
                    if not self.G.has_edge(u, n):
                        self._add_rail_edge(u, n)
                    if not self.G.has_edge(n, v):
                        self._add_rail_edge(n, v)
                    changed = True
                    break

        # (b) segment interior crossings
        changed = True
        while changed:
            changed = False
            edges = self._unique_rail_edges()
            for i in range(len(edges)):
                a, b = edges[i]
                for j in range(i + 1, len(edges)):
                    c, d = edges[j]
                    if len({a, b, c, d}) < 4:
                        continue
                    X = self._seg_intersection(self.nodes[a], self.nodes[b],
                                               self.nodes[c], self.nodes[d])
                    if X is None:
                        continue
                    xid = self._junction_node(X)
                    for p, q in ((a, b), (c, d)):
                        self._remove_rail_edge(p, q)
                        if xid != p and not self.G.has_edge(p, xid):
                            self._add_rail_edge(p, xid)
                        if xid != q and not self.G.has_edge(xid, q):
                            self._add_rail_edge(xid, q)
                    changed = True
                    break
                if changed:
                    break

    def _build_railway(self):
        """Build the sparse rail graph (junction nodes only), then join roads/dests."""
        for a, b in self.rail_lines:
            ua = self._junction_node(a)
            ub = self._junction_node(b)
            if ua != ub:
                self._add_rail_edge(ua, ub)

        # Robustly weld near-miss junctions before wiring roads/destinations.
        self._weld_railway()

        # Join every waypoint and destination to the railway by its perpendicular foot
        # (splitting the nearest rail segment at the foot). Penalised (road cost).
        for rid in list(self.nav_ids) + list(self.dest_ids):
            self._attach_to_rail(rid, self._road_mult)

    def _attach_to_rail(self, ext_id, road_mult, ops=None):
        """
        Connect external node ext_id to the railway via the perpendicular foot on the
        nearest visible rail segment, splitting that segment at the foot.

        If ops is a dict, records the mutation so it can be undone (for temporary
        query-time nodes such as the current pose).
        Returns the foot node id, or None if no rail segment is visible.
        """
        p = self.nodes[ext_id]
        best = None  # (dist, u, v, foot)
        for u, v in self._unique_rail_edges():
            foot, _t = self._proj_seg(p, self.nodes[u], self.nodes[v])
            if self._visible(p, foot):
                d = self._euclidean(p, foot)
                if best is None or d < best[0]:
                    best = (d, u, v, foot)
        if best is None:
            return None

        d, u, v, foot = best
        # Reuse an endpoint if the foot lands on it; else split the segment.
        if self._euclidean(foot, self.nodes[u]) < 1e-6:
            fid = u
        elif self._euclidean(foot, self.nodes[v]) < 1e-6:
            fid = v
        else:
            fid = self._new_rail_node(foot)
            self._remove_rail_edge(u, v)
            self._add_rail_edge(u, fid)
            self._add_rail_edge(fid, v)
            if ops is not None:
                ops["nodes"].append(fid)
                ops["removed_rail"].append((u, v))
                ops["added_rail"] += [(u, fid), (fid, v)]

        w = self._euclidean(p, self.nodes[fid]) * road_mult
        self.G.add_edge(ext_id, fid, weight=w, kind="road")
        self.G.add_edge(fid, ext_id, weight=w, kind="road")
        if ops is not None:
            ops["edges"] += [(ext_id, fid), (fid, ext_id)]
        return fid

    def _undo_attach(self, ops):
        for a, b in ops["edges"]:
            if self.G.has_edge(a, b):
                self.G.remove_edge(a, b)
        for a, b in ops["added_rail"]:
            self._remove_rail_edge(a, b)
        for nid in ops["nodes"]:
            if self.G.has_node(nid):
                self.G.remove_node(nid)
            self.nodes.pop(nid, None)
            self.labels.pop(nid, None)
            self.group_ids.pop(nid, None)
            if nid in self.rail_node_ids:
                self.rail_node_ids.remove(nid)
        for u, v in ops["removed_rail"]:
            self._add_rail_edge(u, v)

    # ------------------------------------------------------------------ queries
    def snap_to_route(self, point, threshold=None):
        if self.route_network is None or self.route_network.is_empty:
            return point
        p = Point(*point)
        if threshold is not None and p.distance(self.route_network) > threshold:
            return point
        snapped, _ = nearest_points(self.route_network, p)
        return (snapped.x, snapped.y)

    def get_route_segments(self):
        # On floors with a hand-drawn railway, only expose the rail skeleton
        # (group 7). The road tier is an O(n^2) visibility mesh kept in the graph
        # for routing/fallback, but drawing it clutters the map and snapping to
        # it is meaningless — on a railway floor the user walks the rail.
        has_rail = bool(getattr(self, "rail_node_ids", None))
        seen = set()
        segs = []
        for u, v in self.G.edges():
            if has_rail and not (
                self.group_ids.get(u) == 7 and self.group_ids.get(v) == 7
            ):
                continue
            key = (min(u, v), max(u, v))
            if key not in seen and u in self.nodes and v in self.nodes:
                seen.add(key)
                p1, p2 = self.nodes[u], self.nodes[v]
                segs.append({"from": list(p1), "to": list(p2)})
        return segs

    def find_path(self, start_id: int, dest_id: int) -> Dict[str, Any]:
        if dest_id not in self.dest_ids:
            return {"error": "Destination must be terminal"}
        if start_id == dest_id:
            return {
                "path_ids": [start_id],
                "path_coords": [self.nodes[start_id]],
                "path_labels": [self.labels[start_id]],
                "total_cost": 0.0,
            }
        try:
            path = nx.dijkstra_path(self.G, source=start_id, target=dest_id)
            coords = [self.nodes[n] for n in path]
            cost = sum(self._euclidean(coords[i], coords[i + 1]) for i in range(len(coords) - 1))
            return {
                "path_ids": path,
                "path_coords": coords,
                "path_labels": [self.labels.get(n, "") for n in path],
                "total_cost": cost,
            }
        except nx.NetworkXNoPath:
            return {"error": "No path found"}

    def _absorb_start_hop(self, raw, heading):
        """
        If the user is already walking down a corridor, drop the short perpendicular
        entry hop so the first instruction is "go forward" instead of a right-angle turn.
        """
        if heading is None:
            return raw
        coords = raw.get("path_coords")
        if not coords or len(coords) < 3:
            return raw
        hop = self._euclidean(coords[0], coords[1])
        if hop >= self.START_HOP_PX:
            return raw
        turn_if_kept = abs(self._norm_angle(self._bearing(coords[0], coords[1]) - heading))
        align_next = abs(self._norm_angle(self._bearing(coords[1], coords[2]) - heading))
        if align_next <= self.START_ALIGN_DEG and turn_if_kept >= self.START_TURN_DEG:
            for k in ("path_coords", "path_ids", "path_labels"):
                if k in raw and len(raw[k]) >= 2:
                    del raw[k][1]
            if "path_coords" in raw:
                raw["total_cost"] = sum(
                    self._euclidean(raw["path_coords"][i], raw["path_coords"][i + 1])
                    for i in range(len(raw["path_coords"]) - 1)
                )
        return raw

    def find_path_from_pose(self, pose_xy, dest_id: int, heading: Optional[float] = None) -> Dict[str, Any]:
        """
        Route from a free-space pose to a destination.

        heading (deg, optional): the user's current facing. When given and the user is
        already aligned with the corridor, the perpendicular entry hop onto the railway
        is absorbed so the first instruction is not a spurious right-angle turn.
        """
        if dest_id not in self.dest_ids:
            return {"error": "Destination must be terminal"}

        vid = -1
        self.nodes[vid] = pose_xy
        self.labels[vid] = "pose"
        self.group_ids[vid] = -1
        self.G.add_node(vid)

        m = self._road_mult
        for nid in self.nav_ids:
            if self._visible(pose_xy, self.nodes[nid]):
                self.G.add_edge(vid, nid, weight=self._euclidean(pose_xy, self.nodes[nid]) * m, kind="road")
        for did in self.dest_ids:
            if self._visible(pose_xy, self.nodes[did]):
                self.G.add_edge(vid, did, weight=self._euclidean(pose_xy, self.nodes[did]) * m, kind="road")

        # Perpendicular entry onto the railway (temporary split, undone afterwards)
        ops = {"nodes": [], "edges": [], "added_rail": [], "removed_rail": []}
        if self.rail_lines:
            self._attach_to_rail(vid, m, ops=ops)

        raw = self.find_path(vid, dest_id)

        if self.rail_lines:
            self._undo_attach(ops)
        if self.G.has_node(vid):
            self.G.remove_node(vid)
        self.nodes.pop(vid, None)
        self.labels.pop(vid, None)
        self.group_ids.pop(vid, None)

        # find_path() already returns the pose as coords[0] (source == vid); just relabel
        # it as the start (do NOT re-insert, or coords[0]==coords[1] duplicates the pose).
        if "path_coords" in raw and raw["path_coords"]:
            raw["path_labels"][0] = "start_pose"
            raw = self._absorb_start_hop(raw, heading)

        return raw

    def get_current_room(self, pose_xy) -> str:
        pt = Point(*pose_xy)
        for poly, lbl in self.room_polygons:
            if poly.contains(pt):
                return lbl or "Unnamed Room"
        return "Unknown"

    def list_all_destinations(self) -> Dict[int, Tuple[str, Tuple[float, float]]]:
        return {d: (self.labels[d], self.nodes[d]) for d in self.dest_ids}

    def get_destination_id_by_name(self, name: str) -> int:
        for d in self.dest_ids:
            if name.lower() in self.labels[d].lower():
                return d
        return None
