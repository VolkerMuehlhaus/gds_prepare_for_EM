# Layers above 200 (EM ports, SUBGND/BACKSIDEGND, REF_FOR_TRANSISTOR, thermal markers, chiplet layers) pass through
# unchanged: gds_prepare_for_EM's command line (it kept only 201..249 before 1.7.1) and run_simplify, the API of
# setupEM's Simplify GDS and gds2palace_mcp, in each of its steps.

import pytest

from conftest import design_summary, run_tool, write_gds
from gds_simplify_for_em import run_simplify

MARKERS = (201, 249, 250, 251, 299, 300, 301, 401, 1201)


def marker_layout(path):
    shapes = [(8, 0, (0, 0, 40, 30)), (126, 0, (0, 0, 40, 30)), (8, 35, (12, 2, 14, 4))]
    shapes += [(layer, 0, (2, 2, 4, 4)) for layer in MARKERS]
    shapes += [(8, 0, (50 + 3 * (n % 6), 3 * (n // 6), 51 + 3 * (n % 6), 1 + 3 * (n // 6))) for n in range(30)]
    return write_gds(path, {"SUB": [(8, 0, (0, 0, 5, 5))], "TOP": shapes + [("ref", "SUB", (0, 0))]})


def test_command_line_keeps_layers_above_200(tmp_path):
    src = marker_layout(tmp_path / "in.gds")
    result = run_tool("gds_prepare_for_EM", src, tmp_path / "out.gds")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    before, after = design_summary(src), design_summary(tmp_path / "out.gds")
    assert [l for l in MARKERS if (l, 0) in after] == list(MARKERS)
    assert all(after[(l, 0)] == before[(l, 0)] for l in MARKERS), "markers unchanged"


@pytest.mark.parametrize("options", [dict(do_cutouts=True), dict(do_floating=True), dict(do_merge=True),
                                     dict(fill_ground_holes=True),
                                     dict(via_layers=[(19, [10], [8])], via_merge_size=1.0)],
                         ids=["cutouts", "floating", "merge", "ground_holes", "via_merge"])
def test_run_simplify_keeps_layers_above_200(tmp_path, options):
    src = marker_layout(tmp_path / "in.gds")
    out = tmp_path / "out.gds"
    run_simplify(src, [8, 10], str(out), cellname="TOP", **options)
    after = design_summary(out)
    assert all((l, 0) in after for l in MARKERS) and (8, 35) in after
