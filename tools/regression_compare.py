# Compare the outputs of two versions of the tools, file by file: for regression checks when the geometry code
# changes (e.g. the gdspy -> gdstk port). Read-only.
#
# Usage: regression_compare [--xor] <dirA> <dirB>
#   Compares every .gds file in dirA with the file of the same name in dirB.
#   Default: per cell and (layer, datatype), the cell's own polygon count and summed polygon area (paths as
#            polygons), hierarchy kept. Fast; exact for flat outputs whose polygons don't overlap and for
#            gds_simplify's hierarchical output.
#   --xor:   the design cell flattened, per (layer, datatype): area of the union and area of the XOR of both
#            versions (gdstk booleans). Shows where geometry moved even if areas balance; slow on millions of
#            polygons.
# A layer whose area agrees to 1e-6 (relative) is reported as equal, also if its polygon count differs: that is
# fracturing (large polygons split at other positions), not a change of the geometry.
# Ends with a summary line; the exit code is the number of files that differ.

import collections
import os
import sys

import gdstk


def design_cell(lib):
    tops = [c for c in lib.top_level() if not c.name.startswith("$$$")] or lib.top_level()

    def area(cell):
        box = cell.bounding_box()
        return 0 if box is None else (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])
    return max(tops, key=area)


def per_cell_sums(path):
    """{cell name: {(layer, datatype): [count, area]}} of each cell's own polygons and paths."""
    out = {}
    for cell in gdstk.read_gds(path).cells:
        sums = collections.defaultdict(lambda: [0, 0.0])
        polygons = list(cell.polygons) + [p for path_ in cell.paths for p in path_.to_polygons()]
        for p in polygons:
            entry = sums[(p.layer, p.datatype)]
            entry[0] += 1
            entry[1] += p.area()
        out[cell.name] = dict(sums)
    return out


def same_area(a, b):
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1.0)


def compare_sums(path_a, path_b):
    a, b = per_cell_sums(path_a), per_cell_sums(path_b)
    lines = []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            lines.append(f"  cell {name} only in {'A' if name in a else 'B'}")
            continue
        for key in sorted(set(a[name]) | set(b[name])):
            (ca, aa), (cb, ab) = a[name].get(key, [0, 0.0]), b[name].get(key, [0, 0.0])
            if not same_area(aa, ab):
                lines.append(f"  {name} {key}: count {ca} -> {cb}  area {aa:.4f} -> {ab:.4f}")
    return lines


def compare_xor(path_a, path_b):
    def by_spec(path):
        shapes = collections.defaultdict(list)
        for p in design_cell(gdstk.read_gds(path)).get_polygons(include_paths=True):
            shapes[(p.layer, p.datatype)].append(p)
        return shapes
    a, b = by_spec(path_a), by_spec(path_b)
    lines = []
    for key in sorted(set(a) | set(b)):
        pa, pb = a.get(key, []), b.get(key, [])
        union_a = sum(p.area() for p in gdstk.boolean(pa, [], "or", precision=1e-4)) if pa else 0.0
        union_b = sum(p.area() for p in gdstk.boolean(pb, [], "or", precision=1e-4)) if pb else 0.0
        xor = sum(p.area() for p in gdstk.boolean(pa, pb, "xor", precision=1e-4))
        if xor > 1e-3:
            lines.append(f"  {key}: count {len(pa)} -> {len(pb)}  area {union_a:.3f} -> {union_b:.3f}  xor {xor:.4f}")
    return lines


def main(argv):
    xor = "--xor" in argv
    args = [a for a in argv if a != "--xor"]
    if len(args) != 2:
        print("Usage: regression_compare [--xor] <dirA> <dirB>")
        return 1
    dir_a, dir_b = args
    files = sorted(f for f in os.listdir(dir_a) if f.endswith(".gds"))
    differing = 0
    for name in files:
        path_b = os.path.join(dir_b, name)
        if not os.path.exists(path_b):
            print(f"{name}: missing in {dir_b}")
            differing += 1
            continue
        lines = (compare_xor if xor else compare_sums)(os.path.join(dir_a, name), path_b)
        print(f"{name}: {'same' if not lines else f'{len(lines)} difference(s)'}", flush=True)
        for line in lines:
            print(line)
        differing += bool(lines)
    print(f"{differing} of {len(files)} file(s) differ")
    return differing


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
