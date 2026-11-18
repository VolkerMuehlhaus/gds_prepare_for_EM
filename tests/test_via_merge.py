# Via array merging: a merged via shape with a hole (a via array around an opening) keeps its full area. A shape with
# holes comes out of the boolean as a keyhole polygon (a zero-width slit from the outline to each hole); shrunk on its
# own, the slit widens and cuts a strip out of the via.

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from gds_prepare_for_EM import merge_via_array_by_metal_overlap


def via_ring(x0, y0, n, size=0.5, pitch=1.0):
    """Vias on the border of an n x n grid: a square ring of vias around an opening."""
    return [np.array([(x0 + i * pitch, y0 + j * pitch), (x0 + i * pitch + size, y0 + j * pitch),
                      (x0 + i * pitch + size, y0 + j * pitch + size), (x0 + i * pitch, y0 + j * pitch + size)])
            for i in range(n) for j in range(n) if i in (0, n - 1) or j in (0, n - 1)]


def merged_shape(vias, above, below, spacing=0.6):
    return unary_union([Polygon(p).buffer(0) for p in merge_via_array_by_metal_overlap(vias, above, below, spacing)])


def test_ring_of_vias_keeps_its_area():
    metal = [np.array([(-1, -1), (12, -1), (12, 12), (-1, 12)], float)]
    shape = merged_shape(via_ring(0, 0, 10), metal, metal)
    expected = box(0, 0, 9.5, 9.5) - box(0.5, 0.5, 9, 9)
    assert shape.symmetric_difference(expected).area < 1e-6, shape.symmetric_difference(expected).area
