# tools/gds_diff: a summary line also for identical files, paths counted, cells with references, --flat for a layout
# against a renamed, flattened result.

import gdspy

import gds_diff
from conftest import write_gds


def test_identical_files_say_so(tmp_path, capsys):
    a = write_gds(tmp_path / "a.gds", {"TOP": [(8, 0, (0, 0, 10, 10))]})
    assert gds_diff.diff(a, a) == 0
    assert capsys.readouterr().out.strip().endswith("no differences in 1 cell(s)")


def test_paths_are_compared(tmp_path, capsys):
    paths = []
    for name, width in (("a", 1.0), ("b", 2.0)):
        lib = gdspy.GdsLibrary()
        cell = gdspy.Cell("TOP", exclude_from_current=True)
        cell.add(gdspy.FlexPath([(0, 0), (10, 0)], width, layer=8))
        lib.add(cell)
        lib.write_gds(str(tmp_path / f"{name}.gds"))
        paths.append(str(tmp_path / f"{name}.gds"))
    assert gds_diff.diff(*paths) == 1
    assert "area 10.000 -> 20.000" in capsys.readouterr().out


def test_cells_with_references_and_flat(tmp_path, capsys):
    hierarchical = write_gds(tmp_path / "h.gds", {"SUB": [(8, 0, (0, 0, 5, 5))],
                                                  "TOP": [(8, 0, (10, 0, 20, 5)), ("ref", "SUB", (0, 0))]})
    flat = write_gds(tmp_path / "f.gds", {"TOP_renamed": [(8, 0, (0, 0, 5, 5)), (8, 0, (10, 0, 20, 5))]})
    assert gds_diff.diff(hierarchical, hierarchical) == 0
    assert gds_diff.diff(hierarchical, flat) > 0, "by name: other cells"
    capsys.readouterr()
    assert gds_diff.diff(hierarchical, flat, flat=True) == 0, "flattened design cells: the same"
