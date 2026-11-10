# Shared helpers for the tests: run a command-line tool of this repo, build small GDS files, summarize a result.
#
# Run from the repo root:   python -m pytest            (all, ~1 min)
#                           python -m pytest -m "not golden"   (without the real test cases, a few seconds)

import os
import subprocess
import sys

import gdspy
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))


@pytest.fixture(autouse=True)
def fresh_gdspy():
    """gdspy registers every new cell in a global library; a fresh one per test avoids "already present" errors."""
    gdspy.current_library = gdspy.GdsLibrary()
    yield


def run_tool(tool, *args, cwd=None):
    """Run <tool>.py of this repo with the arguments; returns the CompletedProcess (text output)."""
    return subprocess.run([sys.executable, os.path.join(REPO, tool + ".py"), *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=900)


def write_gds(path, cells, top_names=None):
    """Write a GDS file. cells: {name: [shapes]}, a shape is (layer, datatype, (x1, y1, x2, y2)) for a rectangle or
    ("ref", cell name, (x, y)) for a placement."""
    lib = gdspy.GdsLibrary()
    made = {name: gdspy.Cell(name, exclude_from_current=True) for name in cells}
    for name, shapes in cells.items():
        for shape in shapes:
            if shape[0] == "ref":
                made[name].add(gdspy.CellReference(made[shape[1]], shape[2]))
            else:
                layer, datatype, (x1, y1, x2, y2) = shape
                made[name].add(gdspy.Rectangle((x1, y1), (x2, y2), layer=layer, datatype=datatype))
    for cell in made.values():
        lib.add(cell)
    lib.write_gds(str(path))
    return str(path)


def design_summary(path):
    """{(layer, datatype): (polygon count, area rounded to 1e-3)} of the design cell of a GDS file (the top-level cell
    with the largest bounding box, not $$$ cells), flattened. Independent of the code under test (gds_diff,
    design_cell), so a test of those cannot pass because of them."""
    lib = gdspy.GdsLibrary(infile=str(path))
    tops = [c for c in lib.top_level() if not c.name.startswith("$$$")]

    def area(cell):
        box = cell.get_bounding_box()
        return 0 if box is None else (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])
    summary = {}
    for (layer, datatype), polygons in max(tops, key=area).get_polygons(by_spec=True).items():
        total = sum(0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
                    for p in polygons)
        summary[(int(layer), int(datatype))] = (len(polygons), round(float(total), 3))
    return summary


def top_cell_names(path):
    return [c.name for c in gdspy.GdsLibrary(infile=str(path)).top_level()]
