# decompose_polygon_holes must give the same shape as the polygon read, also when the slit to a hole runs along the
# hole edge (as KLayout writes it) and passes the root of a strip that reaches into the hole.

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from gds_geometry_utils import decompose_polygon_holes

# a ground plane with two slots and, in each slot, a strip that reaches in from the plane edge
PLANE_WITH_STRIPS = [
    (-427.72, -750.0), (-427.72, -50.0), (458.5, -50.0), (458.5, -239.5), (438.5, -239.5), (438.5, -50.0),
    (0.0, -50.0), (0.0, -450.0), (800.0, -450.0), (800.0, -50.0), (-427.72, -50.0), (-427.72, 450.0), (0.0, 450.0),
    (0.0, 50.0), (438.5, 50.0), (438.5, 239.5), (458.5, 239.5), (458.5, 50.0), (800.0, 50.0), (800.0, 450.0),
    (-427.72, 450.0), (-427.72, 750.0), (1250.0, 750.0), (1250.0, -750.0)]


def test_slit_along_hole_edge_keeps_strip():
    expected = (box(-427.72, -750, 1250, 750) - box(0, 50, 800, 450) - box(0, -450, 800, -50)
                ).union(box(438.5, 50, 458.5, 239.5)).union(box(438.5, -239.5, 458.5, -50))
    decomp = decompose_polygon_holes(PLANE_WITH_STRIPS)
    assert decomp is not None
    assert len(decomp["holes"]) == 2
    shape = Polygon(decomp["exterior_coords"]) - unary_union(decomp["holes"])
    assert shape.symmetric_difference(expected).area < 1e-6
