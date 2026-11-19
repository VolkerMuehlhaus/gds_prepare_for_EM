# Periphery ring detection: a hollow frame near the design edge (a seal ring) is deleted, a circuit block is not, also
# when its wiring is drawn as paths and only fill sits along its border. Paths of the cell itself were left out of the
# coverage before, so such a block (the amplifier stages of FMD_QNC_Stacked_180GHz_PA) looked hollow and lost its
# transistors and via stacks.

import gdstk

from gds_geometry_utils import detect_periphery_ring_cells


def frame(x0, y0, x1, y1, width, layer=8):
    return [gdstk.rectangle((x0, y0), (x1, y0 + width), layer=layer),
            gdstk.rectangle((x0, y1 - width), (x1, y1), layer=layer),
            gdstk.rectangle((x0, y0), (x0 + width, y1), layer=layer),
            gdstk.rectangle((x1 - width, y0), (x1, y1), layer=layer)]


def layout(stage_wiring):
    lib = gdstk.Library()
    seal, stage, top = lib.new_cell("SEAL"), lib.new_cell("STAGE"), lib.new_cell("TOP")
    seal.add(*frame(0, 0, 100, 100, 2))
    stage.add(*frame(5, 5, 65, 95, 1))          # fill along the border of the block
    stage.add(*stage_wiring)
    top.add(gdstk.Reference(seal), gdstk.Reference(stage))
    return lib


def test_seal_ring_found_block_with_path_wiring_kept():
    wiring = [gdstk.FlexPath([(35, 6), (35, 94)], 12, layer=10)]
    found = {c["cell_name"] for c in detect_periphery_ring_cells(layout(wiring), "TOP")}
    assert found == {"SEAL"}


def test_hollow_block_is_still_a_candidate():
    found = {c["cell_name"] for c in detect_periphery_ring_cells(layout([]), "TOP")}
    assert found == {"SEAL", "STAGE"}


def test_hollow_fraction_exact_below_threshold_early_stop_above():
    """The coverage of the core is exact up to stop_above; above it, the batched computation may stop early with a
    lower bound that is still above it (a pad frame with millions of fill polygons took minutes before)."""
    import time
    from gds_geometry_utils import hollow_core_fraction

    bbox = ((0, 0), (100, 100))                   # core: 15..85 in x and y, 4900 um2
    ring = frame(0, 0, 100, 100, 2)
    assert hollow_core_fraction(ring, bbox) == 0.0
    sparse = ring + [gdstk.rectangle((20, 20), (30, 30))]          # 100 um2 of the core: 2 %
    exact = hollow_core_fraction(sparse, bbox)
    assert abs(exact - 100 / 4900) < 1e-9
    assert hollow_core_fraction(sparse, bbox, stop_above=0.05, batch_size=2) == exact

    fill = [gdstk.rectangle((i + d, j + d), (i + d + 0.3, j + d + 0.3))
            for i in range(15, 85) for j in range(15, 85) for d in (0.0, 0.5)]      # 9800 squares, 18 % of the core
    full = hollow_core_fraction(fill, bbox)
    assert abs(full - 9800 * 0.09 / 4900) < 1e-6
    t = time.time()
    early = hollow_core_fraction(fill, bbox, stop_above=0.05, batch_size=1000)
    assert 0.05 < early <= full and time.time() - t < 5


def test_hollow_fraction_ignores_polygons_invalid_in_shapely():
    """As before the gdstk port of this test: a self-intersecting (bow tie) polygon doesn't count."""
    from gds_geometry_utils import hollow_core_fraction

    bbox = ((0, 0), (100, 100))
    bowtie = gdstk.Polygon([(20, 20), (80, 80), (80, 20), (20, 80)])
    assert hollow_core_fraction(frame(0, 0, 100, 100, 2) + [bowtie], bbox) == 0.0
