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
