from shapely.geometry import Point
from shapely.ops import nearest_points
import math
from typing import Tuple

def snap_inside_walkable(
    point: Tuple[float, float],
    walkable_union,
    inward_offset: float = 20.0
) -> Tuple[float, float]:
    """
    Snap a point into walkable space, and pull it slightly inward
    to avoid hugging the boundary.

    Args:
        point: (x, y) to snap.
        walkable_union: Shapely Polygon or MultiPolygon.
        inward_offset: pixels to move inward from boundary.

    Returns:
        (x, y) safely inside walkable area.
    """
    p = Point(*point)
    if walkable_union.contains(p):
        return point

    nearest_geom, _ = nearest_points(walkable_union, p)

    # Compute direction from snapped point to original point
    dx = point[0] - nearest_geom.x
    dy = point[1] - nearest_geom.y
    norm = math.hypot(dx, dy)
    if norm == 0:
        # If directly projected onto boundary, choose random inward bump
        dx, dy = 1.0, 0.0
        norm = 1.0

    # Move inward along opposite direction
    inward_x = nearest_geom.x - (dx / norm) * inward_offset
    inward_y = nearest_geom.y - (dy / norm) * inward_offset

    inward_point = Point(inward_x, inward_y)

    if walkable_union.contains(inward_point):
        return (inward_x, inward_y)
    else:
        # fallback to original nearest if too far inward
        return (nearest_geom.x, nearest_geom.y)
