# STEP 1, cutout filling: only density-fill cutouts are filled. A ground plane's clearance around a circuit is a hole
# too; filling it covered the circuit with metal (PA9203: TopMetal2 became one solid rectangle). So is a one-off
# opening under a pad. Filled: empty holes up to SMALL_HOLE_AREA or of a size that repeats, and holes of a fill tile
# (placed many times) that hold the tile's own fill square. Kept: holes with circuit metal (an island not connected to
# the polygon around the hole), one-off large openings.

import gdstk

import gds_prepare_for_EM as prep

METAL = 134          # TopMetal2


def plane_with_holes(holes, size=200.0, layer=METAL):
    """A square ground plane with rectangular holes, as one polygon (gdstk encodes the holes as keyholes)."""
    plane = gdstk.rectangle((0, 0), (size, size), layer=layer)
    cut = [gdstk.rectangle(a, b) for a, b in holes]
    return gdstk.boolean(plane, cut, "not", layer=layer)


def run(lib):
    design = next(c for c in lib.top_level())
    prep.remove_cutout_keep_hierarchy(lib, [METAL], design_bbox=None, design=design, repeat_count=20)
    return design


def area(cell, layer=METAL):
    return sum(p.area() for p in cell.get_polygons(include_paths=True) if p.layer == layer)


def test_clearance_around_circuit_is_kept_small_empty_hole_filled():
    lib = gdstk.Library()
    top = lib.new_cell("TOP")
    top.add(*plane_with_holes([((20, 20), (120, 120)),          # clearance with a line inside
                               ((150, 150), (154, 154))]))       # 16 um2, empty: fill cutout
    circuit = lib.new_cell("CIRCUIT")
    circuit.add(gdstk.rectangle((30, 60), (110, 70), layer=METAL))
    top.add(gdstk.Reference(circuit))
    before = area(top)
    run(lib)
    # the small hole is filled, the clearance stays (a line of 80 x 10 inside it, not touching the plane)
    assert abs(area(top) - (before + 16)) < 1e-6


def test_one_off_large_empty_opening_is_kept_repeated_ones_filled():
    lib = gdstk.Library()
    top = lib.new_cell("TOP")
    top.add(*plane_with_holes([((50, 50), (130, 130))]))                       # 80 x 80 under a pad
    grid = lib.new_cell("GRID")                                               # a mesh tile: 20 x 20, 14 x 14 hole
    grid.add(*plane_with_holes([((3, 3), (17, 17))], size=20))
    top.add(gdstk.Reference(grid, (300, 0), columns=5, rows=5, spacing=(20, 20)))
    before = area(top)
    run(lib)
    assert abs(area(top) - (before + 25 * 196)) < 1e-6          # the 25 tile holes filled, the pad opening kept


def test_fill_tile_with_its_own_square_inside_is_filled():
    """Filled_Grid tiles: a fill square in the opening of each tile, the tile placed many times."""
    lib = gdstk.Library()
    top = lib.new_cell("TOP")
    tile = lib.new_cell("FILLED_GRID")
    tile.add(*plane_with_holes([((3, 3), (17, 17))], size=20))
    tile.add(gdstk.rectangle((8, 8), (12, 12), layer=METAL))
    top.add(gdstk.Reference(tile, (0, 0), columns=5, rows=5, spacing=(20, 20)))
    before = area(top)
    run(lib)
    # each hole filled: 196 um2 more per tile (the fill square inside overlaps the filled outline, area counted twice)
    assert abs(area(top) - (before + 25 * 196)) < 1e-6


def test_fill_tile_hole_with_foreign_island_is_kept_connected_metal_is_no_island():
    """A repeated tile whose opening holds a pad of the circuit (not connected to the tile) keeps its holes; metal
    that only crosses into the opening from the tile itself is connected, and doesn't stop the filling."""
    for pad, filled in ((((305, 5), (315, 15)), False), (((300, 9), (310, 11)), True)):
        lib = gdstk.Library()
        top = lib.new_cell("TOP")
        tile = lib.new_cell("GRID")
        tile.add(*plane_with_holes([((3, 3), (17, 17))], size=20))
        top.add(gdstk.Reference(tile, (300, 0), columns=5, rows=5, spacing=(20, 20)))
        top.add(gdstk.rectangle(*pad, layer=METAL))     # first: an island in the corner tile's opening
        before = area(top)
        run(lib)
        added = area(top) - before
        assert abs(added - (25 * 196 if filled else 0)) < 1e-6, (pad, added)
