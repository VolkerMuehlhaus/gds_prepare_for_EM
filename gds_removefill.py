# Simplify layouts in IHP SG13G2 technology for EM: 
# remove unconnected dummy metal fill from drawing purpose (data type 0)

import argparse
import gdspy
from collections import defaultdict
from rtree import index  # pip install rtree
import numpy as np


# Layer number <-> name mapping. To adapt this tool to a different
# PDK/layer stack, edit this table - via_layers_list below derives from it
# automatically.
LAYER_NAMES = {
  6: "Cont",
  9: "Passiv",
  19: "Via1",
  29: "Via2",
  41: "Pillar",
  49: "Via3",
  66: "Via4",
  125: "TopVia1",
  129: "Vmim",
  133: "TopVia2",
}
NAME_TO_LAYER = {name: layer for layer, name in LAYER_NAMES.items()}

# Via (and via-like) layers to be excluded from floating polygon removal
VIA_LAYER_NAMES = [
  "Cont", "Via1", "Via2", "Via3", "Via4", "Vmim", "TopVia1", "TopVia2",
  "Pillar", "Passiv",
]
via_layers_list = [NAME_TO_LAYER[name] for name in VIA_LAYER_NAMES]


# --------------------------
# Helper functions
# --------------------------

def create_polygon(points, layer=0, datatype=0, decimals=9):
    """Create a gdspy.Polygon with coordinates rounded to avoid floating-point issues."""
    rounded = np.round(points, decimals)
    return gdspy.Polygon(rounded,layer=layer,datatype=datatype)

def bbox_size(poly, tol=1e-6):
    """Return width and height of polygon bounding box rounded to tolerance."""
    (xmin, ymin), (xmax, ymax) = poly.get_bounding_box()
    w = round((xmax - xmin) / tol) * tol
    h = round((ymax - ymin) / tol) * tol
    return (w, h)

def touches_any(poly, rtree_idx, all_polys, precision=1e-5):
    """
    Return True if poly touches or intersects any other polygon in all_polys.

    Any touching neighbor counts as a real connection, including a
    same-size one: identical-size polygons that touch each other must not
    be treated as fill and deleted, since real connected metal is often
    built from many touching copies of the same unit tile too, same as
    dummy fill is. To correctly tell a genuinely floating cluster of
    touching same-size fill units apart from real connected metal made of
    the same-size tiles, run merge_polygons_by_layer() on the cell
    *before* calling find_isolated_same_size_polygons_by_layer(): merging
    fuses each connected cluster into a single polygon first, so the
    isolation test here only ever has to ask "does this (already-merged)
    shape touch anything else" - which is unambiguous.
    """
    (xmin, ymin), (xmax, ymax) = poly.get_bounding_box()

    candidate_ids = list(rtree_idx.intersection((xmin, ymin, xmax, ymax)))

    # oversize for a small overlap with possible neighbors
    offset = 0.01
    oversizepoly=gdspy.offset(poly, offset, join='miter', tolerance=2, precision=0.001, join_first=True, max_points=1999)
    for cid in candidate_ids:
        other = all_polys[cid]
        if other is poly:
            continue
        if gdspy.boolean(oversizepoly, other, 'and', precision=precision) is not None:
            return True  # touches another polygon
    return False  # isolated


def merge_polygons(polygons):
    """Merge (boolean OR) a list of LPPpolylist-style polygons, return the merged point-arrays."""
    mergedpolygonset = gdspy.boolean(polygons, None, "or", max_points=999)
    return mergedpolygonset.polygons


def merge_polygons_by_layer(cell):
    """
    Merge (boolean OR) all polygons within each (layer, datatype) group of
    a flat cell into the minimal set of merged polygons. Must run before
    find_isolated_same_size_polygons_by_layer(), see touches_any() for why.
    """
    new_lib = gdspy.GdsLibrary()
    new_cell = gdspy.Cell(cell.name + "_merged_by_layer")

    polys_by_layer = cell.get_polygons(by_spec=True, depth=0)
    for (layer, datatype), polys in polys_by_layer.items():
        if not polys:
            continue
        merged_points = merge_polygons(polys)
        for pts in merged_points:
            new_cell.add(gdspy.Polygon(pts, layer=layer, datatype=datatype))

    new_lib.add(new_cell)
    return new_lib


def find_isolated_same_size_polygons_by_layer(cell, via_layers_list, minsize=1, maxsize=None, mincount=20, size_tol=1e-6 ):
    """
    Flatten the hierarchy and remove isolated (floating) polygons that
    repeat with the same bounding-box size at least `mincount` times on a
    layer - this is the signature of auto-generated or man-made dummy
    metal fill. A same-size group with fewer than `mincount` members is
    left untouched even if isolated, since it's more likely to be
    deliberate design content than repetitive fill.

    IMPORTANT: run merge_polygons_by_layer() on `cell` before calling
    this - see touches_any() for why.

    Gating on repeat count (rather than a fixed absolute size ceiling)
    generalizes across layers/layouts automatically: a size that
    legitimately repeats often enough is treated as fill regardless of
    its absolute micron size, and groups below mincount are skipped
    before running the (relatively expensive) isolation test at all.
    `maxsize` remains available as an optional manual cap; leave it as
    None to rely on mincount alone (the default).

    Only considers polygons on the same layer for isolation.
    """
    # create new library with new cell to hold polygons that are NOT removed
    new_lib = gdspy.GdsLibrary()
    new_cell = gdspy.Cell(cell.name + "_cleaned")

    # Flatten hierarchy
    flat = cell.flatten()

    # Get polygons grouped by layer
    polys_by_layer = flat.get_polygons(by_spec=True)  # returns dict {(layer, datatype): [polys]}
    isolated_per_layer = defaultdict(dict)

    for (layer, datatype), polys_raw in polys_by_layer.items():
        if len(polys_raw) == 0:
            continue

        # Convert to gdspy.Polygon and remove exact duplicates (including layer and datatype)
        raw_polys = [create_polygon(p, layer, datatype) for p in polys_raw]
        seen = set()
        all_polys = []

        for poly in raw_polys:
            a = poly
            pts = poly.polygons[0]
            if pts is None or pts.size == 0:
                continue
            # Hashable signature includes points, layer, and datatype
            sig = (layer, datatype, tuple(map(tuple, np.round(pts, 9))))
            if sig not in seen:
                seen.add(sig)
                all_polys.append(poly)

        # only check if not via layer
        if layer not in via_layers_list:

            groups = defaultdict(list)
            for poly in all_polys:
                groups[bbox_size(poly, tol=size_tol)].append(poly)

            # Build R-tree for the whole layer once, used by any size group
            # that passes the cheap pre-filters below
            rtree_idx = index.Index()
            for i, poly in enumerate(all_polys):
                (xmin, ymin), (xmax, ymax) = poly.get_bounding_box()
                rtree_idx.insert(i, (xmin, ymin, xmax, ymax))

            isolated = {}
            for size, polys in groups.items():
                w, h = size

                # cheap pre-filters, avoid the isolation test entirely when
                # the group can't possibly qualify for removal
                below_mincount = len(polys) < mincount
                below_minsize = w < minsize or h < minsize
                above_maxsize = maxsize is not None and (w > maxsize or h > maxsize)
                if below_mincount or below_minsize or above_maxsize:
                    for poly in polys:
                        new_cell.add(poly)
                    continue

                keep = []
                for poly in polys:
                    if touches_any(poly, rtree_idx, all_polys, precision=1e-5):
                        new_cell.add(poly)
                    else:
                        keep.append(poly)

                if len(keep) >= mincount:
                    isolated[size] = keep
                else:
                    # not enough of them actually turned out isolated - keep them all
                    for poly in keep:
                        new_cell.add(poly)

            if isolated:
                isolated_per_layer[layer] = isolated
        else:
            # via layer, append with no changes
            for poly in all_polys:
                 new_cell.add(poly)

    for layer, layer_data in isolated_per_layer.items():
        print(f"Layer {layer}:")
        for size, polys in layer_data.items():
            print(f"  Size {size}: removed {len(polys)} isolated polygons")

    new_lib.add(new_cell)
    return new_lib




# --------------------------
#   main
# --------------------------


def print_run_config(parser, args):
    """Print the full set of available commandline options and how to use
    them, followed by the value actually used for each on this run - so a
    user never has to guess what a run did after the fact."""
    print(parser.format_help())
    print("Resolved configuration for this run:")
    for name, value in sorted(vars(args).items()):
        print(f"  {name} = {value}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Remove unconnected dummy metal fill from IHP SG13G2 GDSII layout.")
    parser.add_argument("input_gds", help="input GDSII file")
    parser.add_argument("output_gds", nargs="?", help="output GDSII file (default: <input>_cleaned.gds)")
    parser.add_argument("--minsize", type=float, default=1,
                         help="minimum polygon bbox size (microns) to consider for floating-fill removal (default: 1)")
    parser.add_argument("--maxsize", type=float, default=None,
                         help="maximum polygon bbox size (microns) to consider for floating-fill removal "
                              "(default: no limit - a same-size group is judged by how often it repeats "
                              "instead, see --mincount)")
    parser.add_argument("--mincount", type=int, default=20,
                         help="minimum number of same-size isolated polygons on a layer before they're "
                              "treated as removable fill (default: 20)")
    args = parser.parse_args()
    print_run_config(parser, args)

    input_name = args.input_gds
    output_name = args.output_gds if args.output_gds else input_name.replace('.gds', '_cleaned.gds')

    print ("Input file: ", input_name)

    lib = gdspy.GdsLibrary()
    lib.read_gds(input_name)
    top = lib.top_level()[0]
    top.flatten()

    # merge touching same-layer polygons first - see touches_any() for why
    # this has to happen before the isolation test
    merged_lib = merge_polygons_by_layer(top)
    merged_top = merged_lib.top_level()[0]

    new_lib = find_isolated_same_size_polygons_by_layer(
        merged_top, via_layers_list, minsize=args.minsize, maxsize=args.maxsize, mincount=args.mincount
    )
    new_lib.write_gds(output_name)

    print('\n\nFINISHED: Created output file ', output_name)


if __name__ == "__main__":
    main()
