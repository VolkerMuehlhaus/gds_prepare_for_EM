# Read-only structural diff between two GDSII files.
# Reports, per cell and per (layer, datatype): polygon count delta,
# vertex-count stats, and total area delta. Used to verify that
# simplification changes have the intended effect without regressing
# unrelated geometry. Does not write any files.
#
# Usage: gds_diff <fileA.gds> <fileB.gds>

import sys
import gdspy


def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    x2 = points[:, 0].take(range(1, len(points) + 1), mode="wrap")
    y2 = points[:, 1].take(range(1, len(points) + 1), mode="wrap")
    return 0.5 * abs((x * y2 - x2 * y).sum())


def load_summary(path):
    """Return {cell_name: {(layer, datatype): {"count": int, "verts": [int], "area": float}}}"""
    lib = gdspy.GdsLibrary(infile=path)
    summary = {}
    for cell in lib.cells.values():
        cell_summary = {}
        for poly in cell.polygons:
            for pts, layer, datatype in zip(poly.polygons, poly.layers, poly.datatypes):
                key = (layer, datatype)
                entry = cell_summary.setdefault(key, {"count": 0, "verts": [], "area": 0.0})
                entry["count"] += 1
                entry["verts"].append(len(pts))
                entry["area"] += polygon_area(pts)
        summary[cell.name] = cell_summary
    return summary


def format_verts(verts):
    if not verts:
        return "n=0"
    return f"n={len(verts)} verts(min/avg/max)={min(verts)}/{sum(verts) / len(verts):.1f}/{max(verts)}"


def diff(path_a, path_b):
    a = load_summary(path_a)
    b = load_summary(path_b)

    all_cells = sorted(set(a) | set(b))
    for cell_name in all_cells:
        a_layers = a.get(cell_name, {})
        b_layers = b.get(cell_name, {})
        all_keys = sorted(set(a_layers) | set(b_layers))
        cell_has_diff = False
        lines = []
        for key in all_keys:
            ea = a_layers.get(key, {"count": 0, "verts": [], "area": 0.0})
            eb = b_layers.get(key, {"count": 0, "verts": [], "area": 0.0})
            if ea["count"] == eb["count"] and abs(ea["area"] - eb["area"]) < 1e-6:
                continue
            cell_has_diff = True
            lines.append(
                f"  layer {key}: count {ea['count']} -> {eb['count']}  "
                f"area {ea['area']:.3f} -> {eb['area']:.3f}  "
                f"({format_verts(ea['verts'])}) -> ({format_verts(eb['verts'])})"
            )
        if cell_has_diff:
            print(f"cell {cell_name}:")
            for line in lines:
                print(line)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: gds_diff <fileA.gds> <fileB.gds>")
        sys.exit(1)
    diff(sys.argv[1], sys.argv[2])
