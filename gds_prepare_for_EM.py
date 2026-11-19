# Prepare IHP SG13G2 GDSII layout for EM simulation: an all-in-one pipeline
# combining cutout removal, periphery ring/frame deletion, via-array
# simplification, round-pad-to-octagon conversion, and floating-fill removal.

########################################################################
#
# Copyright 2025 Volker Muehlhaus and IHP PDK Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
########################################################################

import argparse
import sys
import time
import gdstk
from collections import defaultdict
from rtree import index  # pip install rtree
import numpy as np
import shapely
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from gds_geometry_utils import (
    decompose_polygon_holes,
    is_circle_like,
    simplify_round_polygon_to_octagon,
    is_ring_candidate,
    detect_and_delete_periphery_rings,
    cut_holes_as_valid_polygons,
    validate_and_repair_polygons,
    design_cell,
    only_cell,
    polygons_by_spec,
    cell_layers,
    flat_polygon_cell,
    Progress,
)

__version__ = "1.8.0"

# a polygon-with-hole where the hole covers this much of the exterior area
# is treated as a thin ring, not "fill with cutout" - it gets deleted
# instead of replaced by a solid shape
RING_HOLE_FRACTION_THRESHOLD = 0.90

# via array merging: vias closer than this (microns) are merged, if they
# connect the same metal shapes above and below. Larger than the largest
# IHP SG13G2 via spacing (TopVia2: 1.06 um), so every regular array merges.
DEFAULT_VIA_MERGE_SIZE = 2.0
VIA_TILE_SIZE = 2000      # vias per tile when growing a large via group (_merge_via_group)


# Step 1: single source of truth for layer number <-> name. To adapt this
# tool to a different PDK/layer stack, edit this table (and VIA_ABOVE_BELOW
# below) - everything else derives from these two.
LAYER_NAMES = {
  1: "Activ",
  6: "Cont",
  8: "Metal1",
  9: "Passiv",
  10: "Metal2",
  19: "Via1",
  29: "Via2",
  30: "Metal3",
  36: "MIM",
  41: "Pillar",
  49: "Via3",
  50: "Metal4",
  66: "Via4",
  67: "Metal5",
  125: "TopVia1",
  126: "TopMetal1",
  129: "Vmim",
  133: "TopVia2",
  134: "TopMetal2",
  157: "LBE",
}
NAME_TO_LAYER = {name: layer for layer, name in LAYER_NAMES.items()}

# Step 2: real vias, by name - which metal layer sits above/below each. This
# alone defines "which named layers are (real) vias" - no separate via-name
# list needed, it's just VIA_ABOVE_BELOW's keys.
VIA_ABOVE_BELOW = {
  "Cont":    ("Metal1", "Activ"),
  "Via1":    ("Metal2", "Metal1"),
  "Via2":    ("Metal3", "Metal2"),
  "Via3":    ("Metal4", "Metal3"),
  "Via4":    ("Metal5", "Metal4"),
  "Vmim":    ("TopMetal1", "MIM"),
  "TopVia1": ("TopMetal1", "Metal5"),
  "TopVia2": ("TopMetal2", "TopMetal1"),
}
# Passiv and Pillar aren't real vias (no above/below - never via-merged) but
# are still via-like for fill-removal purposes: excluded from floating-fill
# removal same as a real via, while also getting cutout/circle detection
# like a metal layer.
EXTRA_VIA_LIKE_NAMES = ["Passiv", "Pillar"]

# only metals from this layer list are included in output file. Everything
# in LAYER_NAMES that isn't a real via (VIA_ABOVE_BELOW) is a metal layer -
# no separate metal-name list needed either.
metal_layers_list = [NAME_TO_LAYER[name] for name in LAYER_NAMES.values() if name not in VIA_ABOVE_BELOW]

# Via layers to be excluded from floating polygon removal
via_layers_list = [NAME_TO_LAYER[name] for name in list(VIA_ABOVE_BELOW) + EXTRA_VIA_LIKE_NAMES]


# layers in this purpose list are EXCLUDED from output file
exclude_purpose_list = [
  28, # noqrc
  22, # filler
  23, # nofill
  21  # block
]

# Layers above the via layer
layer_above_dict = {NAME_TO_LAYER[via]: NAME_TO_LAYER[above] for via, (above, below) in VIA_ABOVE_BELOW.items()}

# Layers below the via layer
layer_below_dict = {NAME_TO_LAYER[via]: NAME_TO_LAYER[below] for via, (above, below) in VIA_ABOVE_BELOW.items()}


# ----------------------------------------------
# Helper functions for floating polygon removal
# ----------------------------------------------

def create_polygon(points, layer=0, datatype=0, decimals=9):
    """Create a gdstk.Polygon with coordinates rounded to avoid floating-point issues."""
    rounded = np.round(points, decimals)
    return gdstk.Polygon(rounded, layer=layer, datatype=datatype)

def bbox_size(poly, tol=1e-6):
    """Return width and height of polygon bounding box rounded to tolerance."""
    (xmin, ymin), (xmax, ymax) = poly.bounding_box()
    w = round((xmax - xmin) / tol) * tol
    h = round((ymax - ymin) / tol) * tol
    return (w, h)

def touches_any(poly, rtree_idx, all_polys, precision=1e-5):
    """
    Return True if poly touches or intersects any other polygon in all_polys.

    Any touching neighbor counts as a real connection, including a
    same-size one: identical-size polygons that touch each other must not
    be treated as fill and deleted, since real connected metal (e.g. a
    Metal1/Metal2 mesh or ground plane) is often built from many touching
    copies of the same unit tile too. To correctly tell a genuinely
    floating cluster of touching same-size fill units apart from real
    connected metal made of the same-size tiles, run
    merge_polygons_by_layer() on the cell *before* calling
    find_isolated_same_size_polygons_by_layer(): merging fuses each
    connected cluster into a single polygon first, so the isolation test
    here only ever has to ask "does this (already-merged) shape touch
    anything else" - which is unambiguous.
    """
    (xmin, ymin), (xmax, ymax) = poly.bounding_box()

    candidates = [all_polys[cid] for cid in rtree_idx.intersection((xmin, ymin, xmax, ymax))]
    candidates = [other for other in candidates if other is not poly]
    if not candidates:
        return False  # isolated, nothing near it

    # oversize for a small overlap with possible neighbors
    offset = 0.01
    oversizepoly = gdstk.offset(poly, offset, join='miter', tolerance=2, precision=0.001, use_union=True)
    for other in candidates:
        if gdstk.boolean(oversizepoly, other, 'and', precision=precision):
            return True  # touches another polygon
    return False  # isolated


def find_isolated_same_size_polygons_by_layer(cell, layers_list, minsize=1, maxsize=None, mincount=20, size_tol=1e-6 ):
    """
    Flatten the hierarchy and remove isolated (floating) polygons that
    repeat with the same bounding-box size at least `mincount` times on a
    layer - this is the signature of auto-generated or man-made dummy
    metal fill. A same-size group with fewer than `mincount` members is
    left untouched even if isolated, since it's more likely to be
    deliberate design content than repetitive fill.

    IMPORTANT: run merge_polygons_by_layer() on `cell` before calling this.
    Isolation here means "touches nothing else at all", including a
    same-size neighbor - identical-size polygons that touch each other
    must not be deleted, since real connected metal (e.g. a Metal1/Metal2
    mesh or ground plane) is frequently built from many touching copies of
    the same unit tile too, same as dummy fill is. The only way to tell
    a genuinely floating cluster of touching same-size fill units apart
    from real connected metal made of the same-size tiles is to merge
    first: merging fuses every connected cluster (fill or real metal)
    into a single polygon, so a floating fill cluster becomes its own
    small isolated shape while a real connected network merges into one
    (or a few) large, usually uniquely-shaped/sized polygon(s) that won't
    match the same-size-repeat-count fill signature. Calling this on
    unmerged geometry is exactly what caused whole real metal layers to
    be wiped out in an earlier version - don't skip the merge step.

    Gating on repeat count (rather than a fixed absolute size ceiling)
    generalizes across layers/layouts automatically: a size that
    legitimately repeats often enough is treated as fill regardless of
    its absolute micron size, and groups below mincount are skipped
    before running the (relatively expensive) isolation test at all -
    so a large one-off shape, which can never reach mincount, is never
    tested. `maxsize` remains available as an optional manual cap for
    cases where you want to exclude large sizes outright regardless of
    repeat count; leave it as None to rely on mincount alone (the
    default, and the recommended setting for most layouts).

    Only considers polygons on the same layer for isolation.
    """
    # create new library with new cell to hold polygons that are NOT removed
    new_lib = gdstk.Library()
    new_cell = gdstk.Cell(cell.name + "_cleaned")

    # Get polygons grouped by layer, hierarchy flattened
    polys_by_layer = polygons_by_spec(cell)  # {(layer, datatype): [point arrays]}
    isolated_per_layer = defaultdict(dict)

    for (layer, datatype), polys_raw in polys_by_layer.items():
        if len(polys_raw) == 0:
            continue

        # Convert to gdstk.Polygon and remove exact duplicates (including layer and datatype)
        raw_polys = [create_polygon(p, layer, datatype) for p in polys_raw]
        seen = set()
        all_polys = []

        for poly in raw_polys:
            pts = poly.points
            if pts is None or pts.size == 0:
                continue
            # Hashable signature includes points, layer, and datatype
            sig = (layer, datatype, tuple(map(tuple, np.round(pts, 9))))
            if sig not in seen:
                seen.add(sig)
                all_polys.append(poly)

        # only check if not via layer
        if layer in layers_list:

            groups = defaultdict(list)
            for poly in all_polys:
                groups[bbox_size(poly, tol=size_tol)].append(poly)

            # Build R-tree for the whole layer once, used by any size group
            # that passes the cheap pre-filters below
            rtree_idx = index.Index()
            for i, poly in enumerate(all_polys):
                (xmin, ymin), (xmax, ymax) = poly.bounding_box()
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
            print(f"    Size ({size[0]:.2f},{size[1]:.2f}): removed {len(polys)} isolated polygons")

    new_lib.add(new_cell)
    return new_lib




   
# ----------------------------------------------
# Helper functions for via array merging
# ----------------------------------------------

def _shapely_polygons(point_arrays):
  """Shapely polygons of a list of point arrays, as a numpy object array (vectorized per vertex count)."""
  out = np.empty(len(point_arrays), dtype=object)
  by_length = defaultdict(list)
  for i, pts in enumerate(point_arrays):
    by_length[len(pts)].append(i)
  for _, idx in by_length.items():
    out[idx] = shapely.polygons(np.stack([point_arrays[i] for i in idx]))
  return out


def _metal_shapes(polygons):
  """The metal shapes of a layer as shapely Polygons: the polygons joined where they touch or overlap.
  merge_polygons() fractures large merged shapes into several polygons; joined again, a via on a fracture line
  belongs to its metal shape, not to whichever piece the fracture happened to put more of it in."""
  parts = [ShapelyPolygon(p).buffer(0) for p in polygons if len(p) >= 3]
  union = unary_union(parts) if parts else None
  if union is None or union.is_empty:
    return []
  return [g for g in getattr(union, 'geoms', [union]) if g.geom_type == 'Polygon' and g.area > 0]


def _best_overlapping_polygons(vias, metals):
  """
  For each via (numpy array of shapely Polygons), the index of the polygon
  in metals that overlaps it the most (by area), or -1 if nothing overlaps
  it. Using actual overlap area (not just centroid containment) correctly
  handles a via that straddles the edge of a non-convex metal shape.
  Vectorized: a via inside one metal shape (almost all of them) is matched
  by a covers test, the overlap area is only computed for the others.
  """
  best = np.full(len(vias), -1)
  if not metals or not len(vias):
    return best
  metals = np.array(metals, dtype=object)
  shapely.prepare(metals)
  # the vias in the tree, the metal shapes as query: a query geometry is prepared once and tested against its
  # candidates. The other way round, each of up to millions of vias was tested against a large metal shape
  # (a ground plane with thousands of vertices) unprepared: minutes on a layout with fill vias.
  metal_idx, via_idx = shapely.STRtree(vias).query(metals, predicate='intersects')
  if not len(via_idx):
    return best
  covered = shapely.covers(metals[metal_idx], vias[via_idx])
  best[via_idx[covered]] = metal_idx[covered]
  rest = ~covered & (best[via_idx] == -1)
  areas = shapely.area(shapely.intersection(metals[metal_idx[rest]], vias[via_idx[rest]]))
  best_area = defaultdict(float)
  for v, m, area in zip(via_idx[rest], metal_idx[rest], areas):
    if area > best_area[v]:
      best_area[v] = area
      best[v] = m
  return best


def _spatial_tiles(polygons, per_tile=VIA_TILE_SIZE):
  """Index lists of the polygons, grouped by a square grid over their centers, about per_tile polygons per grid
  cell. Up to per_tile polygons: one list of all, in their order."""
  if len(polygons) <= per_tile:
    return [list(range(len(polygons)))]
  centers = np.array([np.asarray(p).mean(axis=0) for p in polygons])
  (x0, y0), (x1, y1) = centers.min(axis=0), centers.max(axis=0)
  cells = max(1, int(np.sqrt(len(polygons) / per_tile)))
  ix = np.minimum(((centers[:, 0] - x0) / max(x1 - x0, 1e-9) * cells).astype(int), cells - 1)
  iy = np.minimum(((centers[:, 1] - y0) / max(y1 - y0, 1e-9) * cells).astype(int), cells - 1)
  key = ix * cells + iy
  order = np.argsort(key, kind="stable")
  return [list(part) for part in np.split(order, np.flatnonzero(np.diff(key[order])) + 1)]


def _merge_via_group(via_polygons, maxspacing):
  """
  Merge vias closer than maxspacing: oversize by half the spacing, OR,
  undersize again. Same method and offset as gds2palace/gds2openEMS
  merge_via_array(), so a given distance means the same in all tools.
  Returns a list of shapely Polygons.
  """
  offset = maxspacing/2 + 0.01
  # use_union, and no fracturing before the shrink: a merged shape with holes is a keyhole polygon (a zero-width
  # slit from the outline to each hole), and a fractured shape is pieces with shared edges. Shrunk on its own, a
  # slit widens and a shared edge opens, each cutting a strip out of the via. Joined first, a slit is a hole again
  # and the pieces are one shape.
  # grown tile by tile: growing distributes over the union (a convex corner of the union is a corner of one via,
  # a concave one the crossing of two grown edges), so the merged result is the same, and Clipper is much faster
  # on many small inputs than on one large one (1.3 million fill vias under a pad frame: 3 s instead of 33 s)
  grown = []
  for tile in _spatial_tiles(via_polygons):
    grown += gdstk.offset([np.asarray(via_polygons[i]) for i in tile], offset, join='miter', tolerance=2,
                          precision=0.001, use_union=True)
  merged = gdstk.boolean(grown, [], "or")
  shrunk = gdstk.offset(merged, -offset, join='miter', tolerance=2, precision=0.001, use_union=True)
  if not shrunk:
    return []
  # fracture: a merged shape with holes is a keyhole ring, which shapely can't read as one polygon
  pieces = [ShapelyPolygon(q.points).buffer(0) for p in shrunk for q in p.fracture(max_points=199)
            if len(q.points) >= 3]
  # fracturing splits one merged shape into several polygons, join them again
  merged_shape = unary_union(pieces)
  return [g for g in getattr(merged_shape, 'geoms', [merged_shape]) if g.geom_type == 'Polygon' and g.area > 0]


def _polygon_to_points(polygon):
  """
  Shapely Polygon -> list of point arrays for GDS. A polygon with holes
  is cut with gdstk, which encodes the holes as keyholes.
  """
  exterior = np.array(polygon.exterior.coords[:-1])
  if not polygon.interiors:
    return [exterior]
  holes = [np.array(ring.coords[:-1]) for ring in polygon.interiors]
  result = gdstk.boolean([exterior], holes, "not")
  return [q.points for p in result for q in p.fracture(max_points=199)]


def merge_via_array_by_metal_overlap(via_polygons, above_polygons, below_polygons,
                                     maxspacing=DEFAULT_VIA_MERGE_SIZE, keep_unmatched=False):
  """
  Replace a via array with a few solid via shapes, without ever connecting
  different metal shapes.

  The vias are grouped by which metal-above polygon and which metal-below
  polygon each one actually lands in. Each group is merged on its own:
  vias closer than maxspacing are joined (oversize by half the spacing,
  OR, undersize). The result is clipped to the overlap of that group's
  metal-above and metal-below polygon.

  - Vias of different metal-above or metal-below polygons are never merged
    together, so a merged via can't short two metal shapes, however large
    maxspacing is.
  - Within a group, only vias closer than maxspacing are joined. Two via
    arrays far apart on the same metal pair (e.g. both ends of a strapped
    Metal1/Metal2 line) stay two shapes; they are not filled in between.
  - A merged via never extends beyond the metal above or below it.

  Vias without metal above or below are dropped, or kept unchanged with
  keep_unmatched=True.

  Returns a list of point arrays (one per merged via shape).
  """
  vias = [np.asarray(v, dtype=float) for v in via_polygons if len(v) >= 3]
  if not vias:
    return []
  above_shapely = _metal_shapes(above_polygons)
  below_shapely = _metal_shapes(below_polygons)

  via_shapely = _shapely_polygons(vias)
  has_area = shapely.area(via_shapely) > 0
  vias = [v for v, keep in zip(vias, has_area) if keep]
  via_shapely = via_shapely[has_area]
  above_of = _best_overlapping_polygons(via_shapely, above_shapely)
  below_of = _best_overlapping_polygons(via_shapely, below_shapely)

  # group vias by which (above-polygon, below-polygon) pair they land in
  pair_vias = defaultdict(list)
  unmatched = []
  for via_pts, above_i, below_i in zip(vias, above_of, below_of):
    if above_i < 0 or below_i < 0:
      unmatched.append(via_pts)  # via without metal on one side
      continue
    pair_vias[(int(above_i), int(below_i))].append(via_pts)

  merged_points = []
  progress = Progress(sum(len(v) for v in pair_vias.values()), "vias merged")
  for (above_i, below_i), vias in pair_vias.items():
    progress.update(len(vias))
    overlap = above_shapely[above_i].intersection(below_shapely[below_i])
    for merged in _merge_via_group(vias, maxspacing):
      clipped = merged.intersection(overlap)
      for part in getattr(clipped, 'geoms', [clipped]):
        if part.geom_type == 'Polygon' and part.area > 0:
          merged_points.extend(_polygon_to_points(part))
  progress.done()
  if keep_unmatched:
    merged_points.extend(unmatched)
  return merged_points


def merge_polygons (polygons, max_points=999):
  """Merge (boolean OR) a list of point arrays, return the merged point arrays.
  Merged shapes with more than max_points vertices are fractured into
  several polygons (as gdspy's boolean did with its max_points).
  """
  polygons = [np.asarray(p) for p in polygons]
  # nothing to merge with a single polygon - skip straight to returning it
  # as-is, which also sidesteps an empty boolean result below for a
  # degenerate/zero-area single shape (e.g. an EM port marker drawn as a
  # zero-width sliver rectangle)
  if len(polygons) <= 1:
    return polygons

  merged = gdstk.boolean(polygons, [], "or")

  # boolean() returns nothing when the resulting union is empty (degenerate
  # input) - keep the original polygons rather than silently dropping them
  if not merged:
    return polygons

  return [q.points for p in merged for q in p.fracture(max_points=max_points)]


def merge_polygons_by_layer(cell, layers_list=None):
  """
  Merge (boolean OR) all polygons within each (layer, datatype) group of a
  flat cell into the minimal set of merged polygons. Reduces polygon count
  where simplification has left many touching/overlapping shapes on the
  same layer (e.g. adjacent solid fill squares) that no longer need to
  stay separate. Only considers polygons directly in `cell` (depth=0),
  which is fine here since by this point in the pipeline the cell is
  already fully flattened.

  layers_list: if given, only layers in this list are merged - every other
  layer's polygons are copied through unchanged into the new cell. Leave as
  None (default, matches the original behavior) to merge every layer found
  in the cell. Restricting to known metal layers also keeps degenerate
  shapes on other layers (e.g. a zero-width via-port marker rectangle)
  exactly as drawn: a boolean would drop them.
  """
  new_lib = gdstk.Library()
  new_cell = gdstk.Cell(cell.name + "_merged_by_layer")

  polys_by_layer = polygons_by_spec(cell, depth=0)
  for (layer, datatype), polys in polys_by_layer.items():
    if not polys:
      continue
    if layers_list is not None and layer not in layers_list:
      new_cell.add(*[gdstk.Polygon(pts, layer=layer, datatype=datatype) for pts in polys])
      continue
    merged_points = merge_polygons(polys)
    new_cell.add(*[gdstk.Polygon(pts, layer=layer, datatype=datatype) for pts in merged_points])

  new_lib.add(new_cell)
  return new_lib


def fill_repeated_holes_by_layer(cell, layers_list, mincount=20, max_hole_area=None, size_tol=1e-3):
  """
  Fill holes that only exist after per-layer merging (STEP 5): a metal mesh
  (ground plane with a regular grid of openings) is often drawn as several
  polygons whose seams cut right through a row of openings. Each piece then
  only has a notch along its edge, not a hole, so remove_cutout_keep_hierarchy()
  (STEP 1, per polygon) can't see it - the full hole only appears once the
  pieces are merged.

  Merging can also enclose a real, one-off opening (e.g. the aperture around
  an antenna), which must stay open. So only holes whose bbox size repeats at
  least `mincount` times on the same layer/datatype are filled - the same
  "repeats often enough to be a pattern" test floating-fill removal uses.
  `max_hole_area`, if given, additionally caps the area of a filled hole.

  Returns a new library; layers without any qualifying holes are copied
  through unchanged.
  """
  new_lib = gdstk.Library()
  # same name as the input cell, so this step leaves no trace in the output
  # top cell name
  new_cell = gdstk.Cell(cell.name)

  polys_by_layer = polygons_by_spec(cell, depth=0)
  for (layer, datatype), polys in polys_by_layer.items():
    fill_holes = []
    if layer in layers_list and polys:
      # union of all pieces (merge_polygons() fractures to max_points, so one
      # hole can be enclosed by several output polygons)
      parts = []
      for pts in polys:
        geom = make_valid(ShapelyPolygon(pts))
        parts.extend(g for g in getattr(geom, 'geoms', [geom]) if g.geom_type == "Polygon" and g.area > 0)
      union = unary_union(parts)
      holes = [ShapelyPolygon(ring) for g in getattr(union, 'geoms', [union]) if g.geom_type == "Polygon"
               for ring in g.interiors]

      holes_by_size = defaultdict(list)
      for hole in holes:
        xmin, ymin, xmax, ymax = hole.bounds
        holes_by_size[(round((xmax - xmin) / size_tol), round((ymax - ymin) / size_tol))].append(hole)

      for (w, h), group in holes_by_size.items():
        if len(group) < mincount:
          continue
        if max_hole_area is not None:
          group = [hole for hole in group if hole.area <= max_hole_area]
        if group:
          print(f'Layer {layer}:{datatype}: filled {len(group)} repeated holes of size ({w * size_tol:g},{h * size_tol:g})')
          fill_holes.extend(group)

    if fill_holes:
      merged_points = merge_polygons(list(polys) + [np.array(hole.exterior.coords)[:-1] for hole in fill_holes])
    else:
      merged_points = polys
    new_cell.add(*[gdstk.Polygon(pts, layer=layer, datatype=datatype) for pts in merged_points])

  new_lib.add(new_cell)
  return new_lib


def _merge_layers_in_place(cell, layers, datatype=0):
    """
    Merge (boolean OR) all polygons on each (layer, datatype) in `layers`
    directly within `cell`, replacing them in place with the merged
    result. No-op for a (layer, datatype) with 0 or 1 polygons.
    """
    polys_by_layer = defaultdict(list)
    for poly in cell.polygons:
        if poly.layer in layers and poly.datatype == datatype:
            polys_by_layer[poly.layer].append(poly)
    for layer, polys in polys_by_layer.items():
        if len(polys) < 2:
            continue
        merged_points = merge_polygons([p.points for p in polys])
        # filter: one pass over the cell (remove() searches the cell once per polygon)
        cell.filter([(layer, datatype)], remove=True, paths=False, labels=False)
        cell.add(*[gdstk.Polygon(pts, layer=layer, datatype=datatype) for pts in merged_points])


def merge_via_arrays_in_cell (input_cell, layers_list, maxspacing=DEFAULT_VIA_MERGE_SIZE):

    # create new library with new cell to hold polygons that are NOT removed
    new_lib = gdstk.Library()
    new_cell = gdstk.Cell(input_cell.name+"_merged")

    # flatten hierarchy below this cell (into a new cell, paths as polygons)
    input_cell = flat_polygon_cell(input_cell)

    # consolidate the metal layers used as via above/below references
    # BEFORE via-array merging runs. A metal shape (e.g. a Metal5 plane
    # under a pad) is very often drawn as many separate, merely-touching
    # fragments (density-fill squares, array-instantiated pieces, etc.)
    # that only become one continuous shape once merged. If via merging
    # runs against those raw fragments instead, it matches individual
    # vias to whichever specific fragment they happen to overlap and
    # artificially splits what is really one connected via cluster into
    # many separate, oddly-shaped output regions - producing exactly the
    # jagged/fragmented via shapes this was supposed to avoid, even
    # though the metal itself is smooth once merged.
    via_metal_layers = set(layer_above_dict.values()) | set(layer_below_dict.values())
    _merge_layers_in_place(input_cell, via_metal_layers)

    # polygons by layer-purpose pair
    LPPpolylist = polygons_by_spec(input_cell, depth=0)
    used_layers = {layer for (layer, purpose) in LPPpolylist}

    for layer_to_extract_gds in layers_list:
        if (layer_to_extract_gds in used_layers):  # use base layer number here to match GDSII

            # go through cells of this layer, to find our target layer
            for LPP in LPPpolylist:
                layer = LPP[0]   
                purpose = LPP[1]
                
                # now get polygons for this one layer-purpose-pair
                if (layer==layer_to_extract_gds) and (purpose not in exclude_purpose_list):
                    layerpolygons = LPPpolylist[(layer, purpose)]
                    numpoly = len(layerpolygons)
                    print(f"Number of polygons on layer {layer}:{purpose}: {numpoly}", flush=True)

                    if layer in layer_above_dict and maxspacing > 0:
                        # via layer: replace the via array with clean solid
                        # shapes covering wherever metal-above and
                        # metal-below actually overlap around a real via
                        layer_above_polygons = LPPpolylist.get((layer_above_dict[layer], 0), [])  # drawing
                        layer_below_polygons = LPPpolylist.get((layer_below_dict[layer], 0), [])  # drawing

                        started = time.monotonic()
                        merged_points = merge_via_array_by_metal_overlap(
                            layerpolygons, layer_above_polygons, layer_below_polygons, maxspacing
                        )
                        print(f"  merged into {len(merged_points)} via polygons "
                              f"({time.monotonic() - started:.1f} s)", flush=True)
                        new_cell.add(*[gdstk.Polygon(pts, layer=layer, datatype=purpose) for pts in merged_points])

                    else:
                        new_cell.add(*[gdstk.Polygon(pts, layer=layer, datatype=purpose) for pts in layerpolygons])
    new_lib.add(new_cell)
    return new_lib


# ----------------------------------------------
# Helper functions for cutout removal
# ----------------------------------------------

def remove_cutout_keep_hierarchy (library, layers_list, design_bbox=None, max_hole_area=None):
  """
  max_hole_area: if given, only cutouts whose total hole area is <=
  max_hole_area get filled in (replaced by a solid outline) - larger holes
  are left untouched. Leave as None (default) to fill every real cutout
  found, regardless of size, matching the original behavior. This only
  gates the "genuine cutout" branch, not thin-ring detection/deletion.
  """
  # iterate over cells
  progress = Progress(len(library.cells), "cells")
  for cell in library.cells:
    progress.update(1)
    # polygons to remove and to add, applied once per cell after the loop:
    # removing polygon by polygon scanned the whole cell each time
    remove, add = [], []

    # iterate over polygons
    for n,poly in enumerate(cell.polygons):
      # points of this polygon
      polypoints = poly.points

      poly_layer = poly.layer
      poly_purpose = poly.datatype


      # -------- Check for polygons with holes (dummy fill with cutout, or thin rings) ---------
      if (poly_layer in layers_list):
        # Polygon of interest, check if we need to process this

        decomp = decompose_polygon_holes(polypoints)
        if decomp is not None:
          is_thin_ring = (
            decomp["hole_area_fraction"] >= RING_HOLE_FRACTION_THRESHOLD
            and design_bbox is not None
            and is_ring_candidate(poly.bounding_box(), design_bbox)
          )

          if is_thin_ring:
            print(cell.name, ' deleting periphery ring polygon #', str(n), 'layer', str(poly_layer))
            remove.append(poly)
          elif max_hole_area is not None and sum(h.area for h in decomp["holes"]) > max_hole_area:
            # cutout is larger than the requested threshold - keep the hole (don't
            # fill it in), but the original polygon's raw "bridge" hole encoding is
            # not guaranteed to be geometrically valid (a solid-body mesher like
            # gds2palace can choke on it) even though max_hole_area only ever meant
            # to gate whether the hole gets FILLED, not whether the polygon is
            # valid - so re-emit it as clean, hole-preserving geometry instead of
            # leaving the original untouched.
            hole_area = sum(h.area for h in decomp["holes"])
            fixed = cut_holes_as_valid_polygons(decomp["exterior_coords"], decomp["holes"],
                                                 poly_layer, poly_purpose)
            if fixed is not None:
              print(cell.name, ' re-encoding oversized cutout polygon #', str(n), 'layer', str(poly_layer),
                    ' as valid geometry (hole area', f'{hole_area:.2f}', 'kept, not filled)')
              remove.append(poly)
              add.extend(fixed)
            else:
              # couldn't find a fracture that validates - fall back to leaving the
              # original alone rather than silently discarding/guessing, but say so
              print(cell.name, ' WARNING: could not re-encode oversized cutout polygon #', str(n),
                    'layer', str(poly_layer), 'as valid geometry - leaving original as-is '
                    '(hole area', f'{hole_area:.2f}', 'exceeds max_hole_area', max_hole_area, ')')
          else:
            # We can be sure we have a dummy shape with cutout.
            print(cell.name, ' replacing cutout polygon #', str(n), 'layer', str(poly_layer))

            # Replace it with a solid version of the exterior outline
            remove.append(poly)
            add.append(gdstk.Polygon(decomp["exterior_coords"], layer=poly_layer, datatype=poly_purpose))
    if remove:
      cell.remove(*remove)
    if add:
      cell.add(*add)
  return library


# ----------------------------------------------
# Helper functions for circle detection 
# ----------------------------------------------

def replace_circles (library, layers_list, min_size=0):
  # iterate over cells
  for cell in library.cells:
    print('Running circle detection on cell ' + str(cell.name))
    remove, add = [], []

    # iterate over polygons
    for n,poly in enumerate(cell.polygons):
      # points of this polygon
      polypoints = poly.points

      poly_layer = poly.layer
      poly_purpose = poly.datatype


      # -------- Check for dummy rectagles with hole inside ---------
      if (poly_layer in layers_list):
        # Polygon of interest, check if we need to process this

        # get number of vertices
        numvertices = len(polypoints)

        # now do the check for circle-like structures
        if numvertices > 11:
          if is_circle_like(polypoints):
              new_points = simplify_round_polygon_to_octagon(polypoints, min_size)
              remove.append(poly)
              add.append(gdstk.Polygon(new_points, layer=poly_layer, datatype=poly_purpose))
    if remove:
      cell.remove(*remove)
    if add:
      cell.add(*add)

  return library


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


class _StepTimer:
    """Prints how long each STEP took, and the total, so a long run shows where the time goes."""

    def __init__(self):
        self.start_time = self.step_time = time.monotonic()
        self.step = None

    def next(self, title):
        self.finish()
        print(title, flush=True)
        self.step = title.split(":")[0].strip()
        self.step_time = time.monotonic()

    def finish(self):
        if self.step is not None:
            print(f"  {self.step} took {time.monotonic() - self.step_time:.1f} s", flush=True)
        self.step = None

    def total(self):
        self.finish()
        return time.monotonic() - self.start_time


def main():
    parser = argparse.ArgumentParser(description="Prepare IHP SG13G2 GDSII layout for EM simulation.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("input_gds", help="input GDSII file")
    parser.add_argument("output_gds", nargs="?", help="output GDSII file (default: <input>_cleaned.gds)")
    parser.add_argument("--fill-minsize", type=float, default=1,
                         help="minimum polygon bbox size (microns) to consider for floating-fill removal (default: 1)")
    parser.add_argument("--fill-maxsize", type=float, default=None,
                         help="maximum polygon bbox size (microns) to consider for floating-fill removal "
                              "(default: no limit - a same-size group is judged by how often it repeats "
                              "instead, see --fill-mincount)")
    parser.add_argument("--fill-mincount", type=int, default=20,
                         help="minimum number of same-size isolated polygons on a layer before they're "
                              "treated as removable fill (default: 20)")
    parser.add_argument("--via-merge-size", type=float, default=DEFAULT_VIA_MERGE_SIZE,
                         help="via array merging: vias closer than this (microns) are merged, if they connect "
                              "the same metal shapes above and below (default: %(default)s, 0 = no via merging)")
    parser.add_argument("--cell", default=None,
                         help="the design cell (default: the top-level cell with the largest bounding box, "
                              "not KLayout's $$$CONTEXT_INFO$$$)")
    parser.add_argument("--max-hole-area", type=float, default=None,
                         help="maximum cutout area (microns squared) to fill in - larger cutouts are left "
                              "untouched (default: no limit - every real cutout found is filled)")
    args = parser.parse_args()
    print_run_config(parser, args)

    input_name = args.input_gds
    output_name = args.output_gds if args.output_gds else input_name.replace('.gds', '_cleaned.gds')

    print ("Input file: ", input_name)
    steps = _StepTimer()

    # All steps run in memory (gdstk). gdspy needed a GDS write+read between
    # steps, its flatten() failed on some in-memory reference structures.
    lib = gdstk.read_gds(input_name)
    unit, precision = lib.unit, lib.precision

    # the design cell, alone with its subcells: every later step takes the first top-level cell
    try:
        design, others = design_cell(lib, args.cell)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    print(f"Cell: {design.name}" + (f" (other top-level cells, left out: {', '.join(others)}; "
                                    "choose another one with --cell)" if others else ""))
    lib = only_cell(lib, design)

    top_cell_name = design.name
    design_bbox = design.bounding_box()

    # STEP 1: remove cutouts in the hierachical design, don't flatten at this stage
    # do this on metal layers (not via layers, not EM port layers)
    hole_area_desc = f'<= {args.max_hole_area}' if args.max_hole_area is not None else 'any size'
    steps.next(f"\nSTEP 1: remove cutouts in the hierachical design ({hole_area_desc}), don't flatten at this stage")
    lib = remove_cutout_keep_hierarchy (lib, metal_layers_list, design_bbox=design_bbox,
                                         max_hole_area=args.max_hole_area)

    # STEP 2: detect and delete thin ring/frame structures on the periphery
    # (e.g. seal rings). This has to run before via array merging, since it
    # needs the design hierarchy intact (a seal ring is typically built
    # from several segment cells, not one polygon-with-hole).
    steps.next(f"\nSTEP 2: detect periphery ring/frame structures (e.g. seal rings)")
    if top_cell_name is not None:
        detect_and_delete_periphery_rings(lib, top_cell_name, apply=True)

    # define layers for processing
    # copy, not alias: extending must not leak vias/port layers into metal_layers_list
    layers_list = list(metal_layers_list)
    layers_list.extend(via_layers_list)
    # in addition to IHP layers, also keep every layer above 200 of the design, unchanged: EM ports (201..299),
    # substrate/backside ground sheets (250, 251), thermal markers (300 and up) and other user layers. Only
    # 201..249 were kept before; a via array merge (STEP 3) writes only the layers of layers_list.
    design_layers = cell_layers(design)
    layers_list.extend(sorted(layer for layer in design_layers if layer > 200 and layer not in layers_list))

    # STEP 3: via array merging, this also flattens the design hierarchy
    steps.next(f'\nSTEP 3: via array merging (distance {args.via_merge_size} um), this also flattens the design hierarchy')
    merged_lib = merge_via_arrays_in_cell (design, layers_list, args.via_merge_size)

    # STEP 4: replace circle-like polygons by octagons. This has to run
    # before per-layer merging (STEP 5): a round pad that is electrically
    # connected to a trace is still its own separate polygon at this point,
    # so its shape reads as circular. If circle detection ran after STEP 5
    # instead, a connected round pad would already be fused with its trace
    # into one non-circular "lollipop" polygon, and circularity detection
    # would (correctly, for that fused shape) no longer recognize it as a
    # circle - so genuinely round, but connected, pads would silently never
    # get simplified. Floating round pads aren't affected either way, but
    # connected ones only stay detectable if this runs first.
    minsize=10
    steps.next(f'\nSTEP 4: replace circle-like polygons by octagons if diameter > {minsize}')
    convert_lib = replace_circles (merged_lib, metal_layers_list, min_size=minsize)

    # STEP 5: merge (boolean OR) polygons within each layer/datatype. This has
    # to run before floating-fill removal (STEP 6): the isolation test there
    # only asks "does this polygon touch anything at all", so two touching
    # copies of the same dummy-fill unit and two touching segments of real
    # connected metal look identical to it unless touching neighbors have
    # already been fused into one shape first. Merging here turns a floating
    # cluster of fill units into its own small isolated polygon, while real
    # connected metal (including the octagons from STEP 4, if they touch a
    # trace) merges into one (or a few) large, usually unique shape.
    steps.next('\nSTEP 5: merge polygons per layer')
    convert_top = convert_lib.top_level()[0]
    merged_by_layer_lib = merge_polygons_by_layer(convert_top, layers_list=layers_list)

    # STEP 5b: fill holes that only appear after merging - a mesh drawn as
    # several polygons whose seams cut through a row of openings leaves only
    # notches per polygon, so STEP 1 can't see them. Only holes whose size
    # repeats >= --fill-mincount times on a layer are filled, so a one-off
    # opening enclosed by merging (e.g. an antenna aperture) stays open.
    hole_area_desc = f', area <= {args.max_hole_area}' if args.max_hole_area is not None else ''
    steps.next(f'\nSTEP 5b: fill holes created by merging, if the same size repeats >= {args.fill_mincount} times per layer{hole_area_desc}')
    merged_top = merged_by_layer_lib.top_level()[0]
    holes_filled_lib = fill_repeated_holes_by_layer(merged_top, metal_layers_list, mincount=args.fill_mincount,
                                                    max_hole_area=args.max_hole_area)

    # STEP 6: remove floating metals that are not connected to anything, if a
    # same size repeats at least --fill-mincount times on a layer. Nothing
    # after this point introduces new touching geometry (removal only
    # removes), so STEP 5's merge is already final - no need to merge again.
    minsize = args.fill_minsize
    maxsize = args.fill_maxsize
    mincount = args.fill_mincount
    size_desc = f'{minsize}..{maxsize}' if maxsize is not None else f'>= {minsize} (no upper limit)'
    steps.next(f'\nSTEP 6: remove floating metals that are not connected to anything, size {size_desc}, repeating >= {mincount} times per size')
    premerged_top = holes_filled_lib.top_level()[0]
    nofloat_lib = find_isolated_same_size_polygons_by_layer(premerged_top, metal_layers_list, minsize=minsize, maxsize=maxsize, mincount=mincount)

    # STEP 7: final validity check/repair, independent of everything above -
    # catches any polygon still geometrically invalid (self-intersecting) no
    # matter the reason (e.g. a cutout STEP 1 deliberately left alone for
    # being larger than --max-hole-area, or anything none of the earlier
    # steps happened to touch at all). A solid-body mesher like gds2palace
    # cannot handle this even though GDSII itself tolerates it, so
    # this always runs, not just when other options are given.
    steps.next('\nSTEP 7: final polygon validity check/repair (always runs)')
    repaired_count, unresolved = validate_and_repair_polygons(nofloat_lib, layers_list)
    if repaired_count:
        print(f'  repaired {repaired_count} invalid polygon(s)')
    if unresolved:
        print(f'  WARNING: {len(unresolved)} polygon(s) could not be auto-repaired - '
              f'these are still geometrically invalid in the output file:')
        for cell_name, n, layer, reason in unresolved:
            print(f'    {cell_name} polygon #{n} layer {layer}: {reason}')
    if not repaired_count and not unresolved:
        print('  no invalid polygons found')

    # SAVE RESULTS, in the unit of the input file. max_points: gdstk would
    # otherwise fracture every polygon over 199 vertices on writing, gdspy didn't
    steps.next("\nWriting the output file")
    nofloat_lib.unit, nofloat_lib.precision = unit, precision
    nofloat_lib.write_gds(output_name, max_points=8190)

    print("Created final output file",output_name)
    print(f"Total time {steps.total():.1f} s")


if __name__ == "__main__":
    main()
