# Simplify layouts in IHP SG13G2 technology for EM: 
# remove unconnected dummy metal fill from drawing purpose (data type 0)

import gdspy
import sys
from collections import defaultdict
from rtree import index  # pip install rtree
import numpy as np


# Via layers to be
via_layers_list = [
  6,  # Cont
  19, # Via1
  29, # Via2
  49, # Via3
  66, # Via4 
  129,# Vmim
  125,# TopVia1
  133, # TopVia2
  41, # dfpad:pillar
  9  # Passiv
]


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

def touches_any(poly, rtree_idx, all_polys, minsize, maxsize, precision=1e-5):
    """
    Return True if poly touches or intersects any other polygon in all_polys.
    Return false if polygon size < minsize or > maxsize
    """
    (xmin, ymin), (xmax, ymax) = poly.get_bounding_box()
    
    size = max(xmax-xmin, ymax-ymin)
    if size<minsize or size>maxsize:
        return True # don't check these, treat as not isolated

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


def find_isolated_same_size_polygons_by_layer(cell, via_layers_list, minsize=1, maxsize=40, size_tol=1e-6 ):
    """
    Flatten the hierarchy and return:
        {layer: {size: [isolated polygons]}}
    Only considers polygons on the same layer for isolation.
    """
    print(f"Removing floating metal with size {minsize} between and {maxsize} units, excluding via layers")


    # create new library with new cell to hold polygons that are NOT removed
    new_lib = gdspy.GdsLibrary()
    new_cell = gdspy.Cell(cell.name)

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

            # Compute sizes
            size_data = []
            for poly in all_polys:
                size = bbox_size(poly, tol=size_tol)
                size_data.append((poly, size))

            # Group by size
            groups = defaultdict(list)
            for poly, size in size_data:
                groups[size].append(poly)

            # Build R-tree for the layer
            rtree_idx = index.Index()
            for i, poly in enumerate(all_polys):
                (xmin, ymin), (xmax, ymax) = poly.get_bounding_box()
                rtree_idx.insert(i, (xmin, ymin, xmax, ymax))

            # Filter isolated polygons
            isolated = {}
            for size, polys in groups.items():
                keep = []
                for poly in polys:
                    if touches_any(poly, rtree_idx, all_polys, minsize=minsize, maxsize=maxsize):
                        new_cell.add(poly)
                    else:    
                        keep.append(poly)
                if keep:
                    isolated[size] = keep

            if isolated:
                isolated_per_layer[layer] = isolated
        else:
            # via layer, append with no changes
            for poly in all_polys:
                 new_cell.add(poly)

    for layer, layer_data in isolated_per_layer.items():
        print(f"Layer {layer}:")
        for size, polys in layer_data.items():
            print(f"  Size {size}: {len(polys)} isolated polygons")

    new_lib.add(new_cell)
    return new_lib




# --------------------------
#   main
# --------------------------



if len(sys.argv) >= 2:
    input_name = sys.argv[1]
    
    # output file specified?
    if len(sys.argv) == 3:
        output_name = sys.argv[2]
    else:
        output_name = input_name.replace('.gds','_cleaned.gds')
    
    print ("Input file: ", input_name)

    lib = gdspy.GdsLibrary()

    lib.read_gds(input_name)
    top = lib.top_level()[0]

    new_lib = find_isolated_same_size_polygons_by_layer(top, via_layers_list, minsize=1, maxsize=40)
    new_lib.write_gds(output_name)


else:
  print ("Usage: gds_removefill <input.gds> [output.gds]")
  