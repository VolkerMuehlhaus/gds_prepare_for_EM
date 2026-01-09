import gdspy
import sys
from collections import defaultdict, Counter
from rtree import index  # pip install rtree
import numpy as np
import math


# only metals from this layer list are included in output file
metal_layers_list = [
  1,
  8,
  9,
  10,
  30,
  36,
  41, 
  50,
  67,
  126,
  134
]


# Via layers to be excluded from floating polygon removal
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


# layers in this purpose list are EXCLUDED from output file
exclude_purpose_list = [
  20, # noqrc
  22, # filler
  23, # nofill   
  32 # block
]

# Layers above the via layer
layer_above_dict = {
  6:  8,  # Cont
  19: 10, # Via1
  29: 30, # Via2
  49: 50, # Via3
  66: 67, # Via4 
  129: 126,# Vmim
  125: 126,# TopVia1
  133: 134 # TopVia2
}

# Layers below the via layer
layer_below_dict = {
  6:  1, # Cont
  19: 8, # Via1
  29: 10, # Via2
  49: 30, # Via3
  66: 50, # Via4 
  129: 36,# Vmim
  125: 67,# TopVia1
  133: 126 # TopVia2
}


# Via spacings to be merged, use large values to TopVia1,TopVia2 to get Pad vias also
via_spacings_dict = {
  6:  1,   # Cont
  19: 1, # Via1
  29: 1, # Via2
  49: 2, # Via3
  66: 4, # Via4 
  129: 2.0,# Vmim
  125: 5.0,# TopVia1, large distance for pads
  133: 7.0 # TopVia2, large distance for pads
}


# ----------------------------------------------
# Helper functions for floating polygon removal
# ----------------------------------------------

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
    
    size_x = xmax-xmin
    size_y = ymax-ymin
    if size_x<minsize or size_y<minsize or size_x>maxsize or size_y>maxsize:
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


def find_isolated_same_size_polygons_by_layer(cell, layers_list, minsize=1, maxsize=40, mincount=20, size_tol=1e-6 ):
    """
    Flatten the hierarchy and return:
        {layer: {size: [isolated polygons]}}
    Only considers polygons on the same layer for isolation.
    """
    # print(f"Removing floating metal with size {minsize} between and {maxsize} units, layers {layers_list}")

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
        if layer in layers_list:

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

            # Now repeat that, and keep polygons with a size that appears nore more than 20 times in isolation on that layer
            for size, polys in groups.items():
                if isolated:
                    # if this size is isolated, check how often it appears
                    isolated_list = isolated.get(size, None)
                    if isolated_list is not None:
                        num_this_size = len(isolated_list)
                        if num_this_size < mincount:
                            for poly in polys:
                                new_cell.add(poly)

        else:
            # via layer, append with no changes
            for poly in all_polys:
                 new_cell.add(poly)

    for layer, layer_data in isolated_per_layer.items():
        print(f"Layer {layer}:")
        for size, polys in layer_data.items():
            if len(polys)>mincount:
                print(f"    Size ({size[0]:.2f},{size[1]:.2f}): found {len(polys)} isolated polygons")

    new_lib.add(new_cell)
    return new_lib




   
# ----------------------------------------------
# Helper functions for via array merging
# ----------------------------------------------

def merge_via_array (polygons, maxspacing):
  """Used internally in processing data from gdspy, does not work on our own all_polygons_list class!

  Args:
      polygons (_type_): LPPpolylist data
      maxspacing (float): offset for oversize/undersize of polygons during via array merge

  Returns:
      _type_: LPPpolylist data
  """

  # Via array merging consists of 3 steps: oversize, merge, undersize
  # Value for oversize depends on via layer
  # Oversized vias touch if each via is oversized by half spacing
  
  offset = maxspacing/2 + 0.01

  if len(polygons)>10:
     precision = maxspacing/10
  else:   
     precision = 0.1

  offsetpolygonset=gdspy.offset(polygons, offset, join='miter', tolerance=2, precision=precision, join_first=True, max_points=999)
  mergedpolygonset=gdspy.boolean(offsetpolygonset, None,"or", max_points=999)
  mergedpolygonset=gdspy.offset(mergedpolygonset, -offset, join='miter', tolerance=2, precision=precision, join_first=False, max_points=999)




  # offset and boolean return PolygonSet, we only need the list of polygons from that
  return mergedpolygonset.polygons 


def merge_polygons (polygons):
  """Used internally to merge layer polygons 

  Args:
      polygons (_type_): LPPpolylist data

  Returns:
      _type_: LPPpolylist data
  """
  mergedpolygonset=gdspy.boolean(polygons, None,"or", max_points=999)

  # offset and boolean return PolygonSet, we only need the list of polygons from that
  return mergedpolygonset.polygons 




def merge_via_arrays_in_cell (input_cell, layers_list):
      
    # create new library with new cell to hold polygons that are NOT removed
    new_lib = gdspy.GdsLibrary()
    new_cell = gdspy.Cell(input_cell.name+"_merged")

    # flatten hierarchy below this cell
    input_cell.flatten(single_layer=None, single_datatype=None, single_texttype=None)

    for layer_to_extract_gds in layers_list:
        # print ("Evaluating layer ", str(layer_to_extract_gds))

        # get layers used in cell
        used_layers = input_cell.get_layers()

        if (layer_to_extract_gds in used_layers):  # use base layer number here to match GDSII
                    
            # iterate over layer-purpose pairs (by_spec=true)
            # do not descend into cell references (depth=0)
            LPPpolylist = input_cell.get_polygons(by_spec=True, depth=0)

            # go through cells of this layer, to find our target layer
            for LPP in LPPpolylist:
                layer = LPP[0]   
                purpose = LPP[1]
                
                # now get polygons for this one layer-purpose-pair
                if (layer==layer_to_extract_gds) and (purpose not in exclude_purpose_list):
                    layerpolygons = LPPpolylist[(layer, purpose)]
                    numpoly = len(layerpolygons)
                    print(f"Number of polygons on layer {layer}:{purpose}: {numpoly}")

                    if layer in via_spacings_dict.keys():
                        # merge via arrays, all other layers skip this step
                        merge_polygon_size = via_spacings_dict.get(layer, 0)
                        layerpolygons = merge_via_array (layerpolygons, merge_polygon_size)

                        # now get polygons on layer above and do boolean and with via
                        layer_num_above = layer_above_dict[layer]
                        layer_above_polygons = LPPpolylist[(layer_num_above, 0)] # drawing

                        # Perform boolean AND with layer above
                        layerpolygons = gdspy.boolean(layerpolygons, layer_above_polygons, operation='and', layer=layer, datatype=purpose)

                        # now get polygons on layer below and do boolean and
                        layer_num_below = layer_below_dict[layer]
                        layer_below_polygons = LPPpolylist[(layer_num_below, 0)] # drawing
                        layerpolygons = gdspy.boolean(layerpolygons, layer_below_polygons, operation='and', layer=layer, datatype=purpose)
                        
                        new_cell.add(layerpolygons)

                    else:
                        # merge polygon layer polygons
                        # layerpolygons = merge_polygons (layerpolygons)

                        for poly in layerpolygons:
                            newpoly = gdspy.Polygon(poly,  layer=layer, datatype=purpose)
                            new_cell.add(newpoly)
    new_lib.add(new_cell)
    return new_lib


# ----------------------------------------------
# Helper functions for cutout removal
# ----------------------------------------------

def remove_cutout_keep_hierarchy (library, layers_list):
  # iterate over cells
  for cell in library:
    # print('cellname = ' + str(cell.name))
  
    # iterate over polygons
    for n,poly in enumerate(cell.polygons):
      # points of this polygon
      polypoints = poly.polygons[0]

      poly_layer = poly.layers[0]
      poly_purpose = poly.datatypes[0]


      # -------- Check for dummy rectagles with hole inside ---------
      if (poly_layer in layers_list):
        # Polygon of interest, check if we need to process this

        # get number of vertices
        numvertices = len(polypoints)

        # criteria for dummy rectangle with cutout:
        # number of vertices = 10 
        # when sorting vertices by distance to center, we have 4 identcal values for outer distance and 4 identical values for inner distance


        # we are interested in polygons with 10 vertices
        if numvertices == 10:
          # get bounding box
          bb = poly.get_bounding_box()

          xmin = bb[0,0]
          ymin = bb[0,1]
          xmax = bb[1,0]
          ymax = bb[1,1]

          xcenter = (xmax+xmin)/2
          ycenter = (ymax+ymin)/2

          # print('      Bounding box xmin=', xmin, ' ymin=', ymin,' xmax=', xmax, ' ymax=', ymax)

          radius_list = []
          for i_vertex in range(numvertices):
            
            # print('polypoints  = ' + str(polypoints))
            x = polypoints[i_vertex][0]
            y = polypoints[i_vertex][1]

            # calculate distance from center
            r = math.sqrt((x-xcenter)**2 + (y-ycenter)**2)
            radius_list.append(r)

          # get count for radius values
          counter = Counter(radius_list)
          sorted_by_count = sorted(counter.items(), key=lambda x: x[1], reverse=True)
          
          r1,n1 = sorted_by_count[0]   
          r2,n2 = sorted_by_count[1]   

          if n1==4 and n2==4:
            # We can be sure we have a dummy square with cutout.
            print(cell.name, ' replacing cutout polygon #', str(n), 'layer', str(poly_layer))      

            # invalidate original polygon
            poly.layers=[0]
            # remove original polygon
            cell.remove_polygons(lambda pts, layer, datatype:layer == 0)

            # Replace it with a solid square   
            basepoly_points=[(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)]    
            basepoly = gdspy.Polygon(basepoly_points, layer=poly_layer, datatype=poly_purpose)
            cell.add(basepoly)     
  return library


# ----------------------------------------------
# Helper functions for circle detection 
# ----------------------------------------------

def replace_circles (library, layers_list, min_size=0):
  # iterate over cells
  for cell in library:
    print('Running circle detection on cell ' + str(cell.name))
  
    # iterate over polygons
    for n,poly in enumerate(cell.polygons):
      # points of this polygon
      polypoints = poly.polygons[0]

      poly_layer = poly.layers[0]
      poly_purpose = poly.datatypes[0]


      # -------- Check for dummy rectagles with hole inside ---------
      if (poly_layer in layers_list):
        # Polygon of interest, check if we need to process this

        # get number of vertices
        numvertices = len(polypoints)

        # now do the check for circle-like structures
        if numvertices > 11:
          if is_circle_like(polypoints):
              new_points = simplify_round_polygon_to_octagon(polypoints, min_size)

              # invalidate original polygon
              poly.layers=[0]
              # remove original polygon
              cell.remove_polygons(lambda pts, layer, datatype:layer == 0)
              basepoly = gdspy.Polygon(new_points, layer=poly_layer, datatype=poly_purpose)
              cell.add(basepoly)     

  return library



def is_circle_like(points, radius_variation_threshold=0.2, min_points=12):
    """
    Detects whether a polygon is circle-like using:
    - Enough points
    - Uniform radii from centroid
    """
    pts = np.asarray(points)

    # Must have enough vertices
    if len(pts) < min_points:
        return False

    # Compute centroid
    center = pts.mean(axis=0)
    
    # Compute radii
    radii = np.linalg.norm(pts - center, axis=1)

    # Radii uniformity (coefficient of variation)
    cv = radii.std() / radii.mean()

    if cv < radius_variation_threshold:
      # Compute distances between consecutive points
      edge_lengths = np.sqrt(np.sum(np.diff(np.vstack([pts, pts[0]]), axis=0)**2, axis=1))
      avg_edge = np.average(edge_lengths)
      max_edge = np.max(edge_lengths)    
      # print(f'edgelength avg {avg_edge} max {max_edge} factor {max_edge/avg_edge}')
      # Check if any edge length differs by more than factor 2
      if (max_edge > 10 * avg_edge) or max_edge > 100:
          return False    

    return cv < radius_variation_threshold



def simplify_round_polygon_to_octagon(points, min_size=0):
    pts = np.asarray(points)

    # --- 1. Estimate circle center ---
    center = pts.mean(axis=0)

    # --- 2. Estimate radius ---
    radii = np.linalg.norm(pts - center, axis=1)
    radius = radii.mean()

    # skip if radius below limit
    if radius < min_size/2:
       return pts

    # --- 3. Fixed angles (rotated by 22.5°) ---
    angles = np.deg2rad(np.arange(0, 360, 45) + 22.5)  # 0,45,90,.. +22.5

    # --- 4. Generate octagon points ---
    octagon = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])

    return octagon


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

    # STEP 1: remove cutouts in the hierachical design, don't flatten at this stage
    # do this on metal layers (not via layers, not EM port layers)
    print(f"\nSTEP 1: remove cutouts in the hierachical design, don't flatten at this stage")
    lib = remove_cutout_keep_hierarchy (lib, metal_layers_list)
    lib.write_gds('tmp.gds')

    # define layers for processing
    layers_list = metal_layers_list
    layers_list.extend(via_layers_list)
    # in addition to IHP layers, also keep layers above 200 that we use for ports etc.
    for layer in range(201,250):
        layers_list.append(layer)

    # STEP 2: via array merging, this also flattens the design hierarchy
    # Read GDSII library
    print(f'\nSTEP 2: via array merging, this also flattens the design hierarchy')
    tmp_library = gdspy.GdsLibrary(infile='tmp.gds')
    top = tmp_library.top_level()[0]
    merged_lib = merge_via_arrays_in_cell (top, layers_list)
    merged_lib.write_gds('merged.gds')

    # STEP 3: remove floating metals that are not connected to anything, with size in a range
    minsize=1
    maxsize=40
    print(f'\nSTEP 3: remove floating metals that are not connected to anything, with size in a range {minsize}..{maxsize}')
    tmp2_library = gdspy.GdsLibrary(infile='merged.gds')
    merged_top = tmp2_library.top_level()[0]
    nofloat_lib = find_isolated_same_size_polygons_by_layer(merged_top, metal_layers_list, minsize=minsize, maxsize=maxsize, mincount=20)
    nofloat_lib.write_gds('cleaned.gds')
    
    # STEP 4: replace circle-like polygons by octagons
    minsize=10
    print(f'\nSTEP 4: replace circle-like polygons by octagons if diameter > {minsize}')
    tmp3_library = gdspy.GdsLibrary(infile='cleaned.gds')
    convert_lib = replace_circles (tmp3_library, metal_layers_list, min_size=10)
    convert_lib.write_gds('converted.gds')

    # SAVE RESULTS
    convert_lib.write_gds(output_name)
    print("Created final output file",output_name)


else:
  print ("Usage: gds_prepare_for_EM <input.gds> [output.gds]")
  