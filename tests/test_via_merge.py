# Via array merging: a merged via shape with a hole (a via array around an opening) keeps its full area. A shape with
# holes comes out of the boolean as a keyhole polygon (a zero-width slit from the outline to each hole); shrunk on its
# own, the slit widens and cuts a strip out of the via. And vias on one metal shape stay one group, also where the
# metal is several polygons (merge_polygons fractures large shapes).

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


def test_metal_in_two_pieces_is_one_shape():
    above = [np.array([(-1, -1), (5, -1), (5, 12), (-1, 12)], float),     # one metal shape drawn as two polygons
             np.array([(5, -1), (12, -1), (12, 12), (5, 12)], float)]
    below = [np.array([(-1, -1), (12, -1), (12, 12), (-1, 12)], float)]
    shape = merged_shape(via_ring(0, 0, 10), above, below)
    assert shape.geom_type == "Polygon" and len(shape.interiors) == 1
    assert abs(shape.area - (9.5 * 9.5 - 8.5 * 8.5)) < 1e-6


def test_large_via_group_merged_in_tiles_gives_the_same_shape(monkeypatch):
    """A large via group is grown tile by tile (much faster in Clipper); the merged shape is the same as grown in
    one piece. Irregular field: rows with gaps, rotated (octagon-like) vias, an opening in the middle."""
    import gds_prepare_for_EM as prep

    rng = np.random.default_rng(1)
    vias = []
    for i in range(70):
        for j in range(70):
            if 30 <= i < 40 and 30 <= j < 40 or rng.random() < 0.1:
                continue
            x, y = i * 1.3 + rng.random() * 0.2, j * 1.1
            if (i + j) % 7 == 0:
                vias.append(np.array([(x + 0.2, y), (x + 0.4, y), (x + 0.6, y + 0.2), (x + 0.6, y + 0.4),
                                      (x + 0.4, y + 0.6), (x + 0.2, y + 0.6), (x, y + 0.4), (x, y + 0.2)]))
            else:
                vias.append(np.array([(x, y), (x + 0.5, y), (x + 0.5, y + 0.5), (x, y + 0.5)]))
    assert len(prep._spatial_tiles(vias, per_tile=200)) > 10
    monkeypatch.setattr(prep._spatial_tiles, "__defaults__", (10 ** 9,))
    one_piece = unary_union(prep._merge_via_group(vias, 0.6))
    monkeypatch.setattr(prep._spatial_tiles, "__defaults__", (200,))
    tiled = unary_union(prep._merge_via_group(vias, 0.6))
    assert one_piece.area > 0 and abs(tiled.area - one_piece.area) < 1e-6 * one_piece.area
    assert one_piece.symmetric_difference(tiled).area < 1e-6
