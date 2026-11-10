# The design cell of a file with several top-level cells: the largest one (never KLayout's $$$CONTEXT_INFO$$$), or
# the one given with --cell. All three command-line tools took the first top-level cell before.

import gdspy
import pytest

from conftest import design_summary, run_tool, top_cell_names, write_gds
from gds_geometry_utils import design_cell, only_cell

CELLS = {"$$$CONTEXT_INFO$$$": [(1, 0, (0, 0, 1000, 1000))],      # largest, but KLayout's: never the design
         "small_device": [(8, 0, (0, 0, 2, 2))],
         "DEVICE": [(8, 0, (0, 0, 3, 3))],
         "DESIGN": [(8, 0, (0, 0, 50, 40)), (126, 0, (5, 5, 45, 35)), (201, 0, (0, 0, 1, 1)), ("ref", "DEVICE", (10, 10))]}


@pytest.fixture
def layout(tmp_path):
    return write_gds(tmp_path / "multi.gds", CELLS)


def test_design_cell(layout):
    lib = gdspy.GdsLibrary(infile=layout)
    cell, others = design_cell(lib)
    assert cell.name == "DESIGN" and sorted(others) == ["small_device"]
    assert design_cell(lib, "small_device")[0].name == "small_device"
    with pytest.raises(ValueError, match="no cell nope"):
        design_cell(lib, "nope")
    alone = only_cell(lib, cell)
    assert sorted(c.name for c in alone.cells.values()) == ["DESIGN", "DEVICE"]


def test_prepare_takes_the_design_cell(layout, tmp_path):
    result = run_tool("gds_prepare_for_EM", layout, tmp_path / "out.gds")
    assert result.returncode == 0, result.stdout[-2000:]
    assert "Cell: DESIGN (other top-level cells, left out: small_device" in result.stdout
    assert len(top_cell_names(tmp_path / "out.gds")) == 1
    assert (126, 0) in design_summary(tmp_path / "out.gds"), "the design's TopMetal1, not a device's metal"


def test_prepare_cell_option(layout, tmp_path):
    result = run_tool("gds_prepare_for_EM", layout, tmp_path / "out.gds", "--cell", "small_device")
    assert result.returncode == 0 and "Cell: small_device" in result.stdout
    assert set(design_summary(tmp_path / "out.gds")) == {(8, 0)}
    result = run_tool("gds_prepare_for_EM", layout, tmp_path / "x.gds", "--cell", "nope")
    assert result.returncode != 0 and "no cell nope in the file" in result.stderr


def test_removefill_and_simplify_take_the_design_cell(layout, tmp_path):
    result = run_tool("gds_removefill", layout, tmp_path / "out.gds")
    assert result.returncode == 0, result.stdout[-2000:]
    assert "Cell: DESIGN" in result.stdout and (126, 0) in design_summary(tmp_path / "out.gds")
    result = run_tool("gds_simplify", layout, cwd=tmp_path)
    assert result.returncode == 0 and "Cell: DESIGN" in result.stdout
