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
    Snap a point into the walkable region, with a slight inward offset
    to avoid being on or near the walkable boundary.

    Args:
        point (Tuple[float, float]): (x, y) coordinate to snap.
        walkable_union: Shapely Polygon or MultiPolygon representing walkable space.
        inward_offset (float): Pixels to move inward from the walkable boundary.

    Returns:
        Tuple[float, float]: (x, y) coordinate safely inside walkable region.
    """
    p = Point(*point)
    if walkable_union.contains(p):
        return point

    # Find the nearest point on the walkable region boundary
    nearest_geom, _ = nearest_points(walkable_union, p)

    # Direction vector from snapped boundary point toward original point
    dx = point[0] - nearest_geom.x
    dy = point[1] - nearest_geom.y
    norm = math.hypot(dx, dy)
    if norm == 0:
        # If the point projects exactly onto the boundary, use a default direction
        dx, dy = 1.0, 0.0
        norm = 1.0

    # Move inward by offset along the reverse direction
    inward_x = nearest_geom.x - (dx / norm) * inward_offset
    inward_y = nearest_geom.y - (dy / norm) * inward_offset

    inward_point = Point(inward_x, inward_y)

    if walkable_union.contains(inward_point):
        return (inward_x, inward_y)
    else:
        # If inward bump goes outside, return the nearest boundary point
        return (nearest_geom.x, nearest_geom.y)
