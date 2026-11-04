# Read-only structural diff between two GDSII files.
# Reports, per cell and per (layer, datatype): polygon count delta,
# vertex-count stats, and total area delta. Used to verify that
# simplification changes have the intended effect without regressing
# unrelated geometry. Does not write any files.
#
# Usage: gds_diff [--flat] <fileA.gds> <fileB.gds>
#   --flat: compare the two design cells flattened (the top-level cell with the
#           largest bounding box in each file), whatever their names - e.g. an
#           input layout against gds_prepare_for_EM's output, whose cell is
#           flattened and renamed. Without it, cells are compared by name.
# Always ends with a summary line, also when nothing differs.

import sys
import gdspy


def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    x2 = points[:, 0].take(range(1, len(points) + 1), mode="wrap")
    y2 = points[:, 1].take(range(1, len(points) + 1), mode="wrap")
    return 0.5 * abs((x * y2 - x2 * y).sum())


def summarize(polygons_by_spec):
    """{(layer, datatype): {"count", "verts", "area"}} from get_polygons(by_spec=True)."""
    summary = {}
    for (layer, datatype), polygons in polygons_by_spec.items():
        entry = summary.setdefault((int(layer), int(datatype)), {"count": 0, "verts": [], "area": 0.0})
        for pts in polygons:
            entry["count"] += 1
            entry["verts"].append(len(pts))
            entry["area"] += polygon_area(pts)
    return summary


def design_cell(lib):
    """The top-level cell with the largest bounding box, not KLayout's $$$CONTEXT_INFO$$$."""
    tops = [c for c in lib.top_level() if not c.name.startswith("$$$")] or lib.top_level()

    def area(cell):
        box = cell.get_bounding_box()
        return 0 if box is None else (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])
    return max(tops, key=area)


def load_summary(path, flat=False):
    """Return {cell_name: {(layer, datatype): {"count": int, "verts": [int], "area": float}}}: each cell's own
    shapes (polygons and paths, without its subcells), or with flat the design cell flattened, under the name
    "(design)"."""
    lib = gdspy.GdsLibrary(infile=path)
    if flat:
        return {"(design)": summarize(design_cell(lib).get_polygons(by_spec=True))}
    return {cell.name: summarize(own_shapes(cell)) for cell in lib.cells.values()}


def own_shapes(cell):
    """{(layer, datatype): [points]} of the cell's own polygons and paths, without its references."""
    shapes = {}
    for poly in cell.polygons:
        for pts, layer, datatype in zip(poly.polygons, poly.layers, poly.datatypes):
            shapes.setdefault((layer, datatype), []).append(pts)
    for path in cell.paths:
        for key, polygons in path.get_polygons(by_spec=True).items():
            shapes.setdefault(key, []).extend(polygons)
    return shapes


def format_verts(verts):
    if not verts:
        return "n=0"
    return f"n={len(verts)} verts(min/avg/max)={min(verts)}/{sum(verts) / len(verts):.1f}/{max(verts)}"


def diff(path_a, path_b, flat=False):
    """Print the differences and a summary line; return the number of (cell, layer) pairs that differ."""
    a = load_summary(path_a, flat)
    b = load_summary(path_b, flat)

    all_cells = sorted(set(a) | set(b))
    differing, differing_cells = 0, 0
    for cell_name in all_cells:
        a_layers = a.get(cell_name, {})
        b_layers = b.get(cell_name, {})
        all_keys = sorted(set(a_layers) | set(b_layers))
        lines = []
        for key in all_keys:
            ea = a_layers.get(key, {"count": 0, "verts": [], "area": 0.0})
            eb = b_layers.get(key, {"count": 0, "verts": [], "area": 0.0})
            if ea["count"] == eb["count"] and abs(ea["area"] - eb["area"]) < 1e-6:
                continue
            lines.append(
                f"  layer {key}: count {ea['count']} -> {eb['count']}  "
                f"area {ea['area']:.3f} -> {eb['area']:.3f}  "
                f"({format_verts(ea['verts'])}) -> ({format_verts(eb['verts'])})"
            )
        if lines:
            where = "" if cell_name in a and cell_name in b else " (only in A)" if cell_name in a else " (only in B)"
            print(f"cell {cell_name}{where}:")
            for line in lines:
                print(line)
            differing += len(lines)
            differing_cells += 1
    if differing:
        print(f"{differing} layer(s) differ in {differing_cells} of {len(all_cells)} cell(s)")
    else:
        print(f"no differences in {len(all_cells)} cell(s)")
    return differing


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--flat"]
    if len(args) != 2:
        print("Usage: gds_diff [--flat] <fileA.gds> <fileB.gds>")
        sys.exit(1)
    diff(args[0], args[1], flat="--flat" in sys.argv[1:])
