# Shared geometry utilities for GDSII layout simplification.
# Used by gds_simplify.py and gds_prepare_for_EM.py.
#
# Covers three generalized simplification steps:
# - decompose_polygon_holes: resolve a polygon-with-hole (GDSII's
#   self-intersecting "bridge" encoding) into exterior + interior rings,
#   for any hole shape/count/rotation - not just an axis-aligned square
#   with a centered square hole.
# - is_circle_like / kasa_circle_fit / simplify_round_polygon_to_octagon:
#   detect near-circular polygons via isoperimetric circularity and fit
#   them with a proper least-squares circle instead of centroid+mean-radius.
# - detect_and_delete_periphery_rings: find thin ring/frame structures
#   near a design's outer boundary (e.g. seal rings), which are commonly
#   built as a hierarchy of segment cells rather than a single polygon
#   with a hole, and delete their geometry.

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

import math

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid


# ============= hole / cutout decomposition =============

def decompose_polygon_holes(points, min_hole_area=1e-6):
    """
    Resolve a GDSII polygon that may encode a hole via a self-intersecting
    'bridge' edge (outer boundary -> bridge -> inner boundary reversed ->
    bridge back) into a clean exterior ring plus interior hole ring(s).

    min_hole_area is an absolute area threshold (design units squared),
    not a fraction of the exterior area: real density-rule cutout holes
    are often a tiny fraction of the fill polygon's area (e.g. a 0.5x0.5
    hole in a 20x20 fill square is 0.06% of the area), so a fractional
    threshold would reject exactly the cutouts we want to catch. The
    default here only filters out true numerical-noise artifacts.

    Returns None if the polygon has no hole (nothing to simplify here),
    otherwise a dict:
      exterior_coords: numpy array of (x, y) points for the outer boundary
      holes: list of shapely Polygon objects, one per hole
      exterior_area: float
      hole_area_fraction: total hole area / exterior area
    """
    try:
        raw = ShapelyPolygon(points)
    except Exception:
        return None

    fixed = raw if raw.is_valid else make_valid(raw)

    if fixed.geom_type == "GeometryCollection":
        # make_valid() on a GDSII "bridge" polygon (outer ring -> zero-width
        # slit -> inner ring -> slit back) typically returns the resolved
        # Polygon-with-hole alongside a degenerate LineString/Point for the
        # slit itself. Keep the polygon part, ignore the lower-dimensional
        # artifacts.
        polygon_parts = [g for g in fixed.geoms if g.geom_type == "Polygon" and g.area > 0]
        if len(polygon_parts) != 1:
            # zero or multiple real polygon parts - too ambiguous to handle
            # generically here, leave the original polygon alone
            return None
        fixed = polygon_parts[0]
    elif fixed.geom_type != "Polygon":
        # e.g. a bowtie that resolves into disjoint parts - too ambiguous
        # to handle generically here, leave the original polygon alone
        return None

    if not fixed.interiors:
        return None

    exterior_area = ShapelyPolygon(fixed.exterior).area
    if exterior_area <= 0:
        return None

    hole_area = sum(ShapelyPolygon(ring).area for ring in fixed.interiors)
    if hole_area < min_hole_area:
        return None  # numerical-noise artifact, not a real hole
    hole_area_fraction = hole_area / exterior_area

    return {
        "exterior_coords": np.array(fixed.exterior.coords),
        "holes": [ShapelyPolygon(ring) for ring in fixed.interiors],
        "exterior_area": exterior_area,
        "hole_area_fraction": hole_area_fraction,
    }


# ============= circle detection =============

def polygon_area_perimeter(points):
    """Shoelace area and perimeter of a simple polygon."""
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    area = 0.5 * abs(np.sum(x * y2 - x2 * y))
    perimeter = float(np.sum(np.sqrt((x2 - x) ** 2 + (y2 - y) ** 2)))
    return area, perimeter


def is_circle_like(points, min_points=12, circularity_threshold=0.90):
    """
    Detect near-circular polygons using isoperimetric circularity:
    4*pi*Area / Perimeter^2, which is 1.0 for a perfect circle and
    decreases for less round shapes (regular octagon ~0.95, hexagon ~0.91,
    square ~0.79). Scale- and rotation-invariant, and more robust than a
    naive radius-uniformity check, which can be fooled by symmetric but
    non-convex point distributions.
    """
    pts = np.asarray(points)
    if len(pts) < min_points:
        return False
    area, perimeter = polygon_area_perimeter(pts)
    if perimeter <= 0:
        return False
    circularity = 4 * math.pi * area / (perimeter ** 2)
    return circularity >= circularity_threshold


def kasa_circle_fit(points):
    """Closed-form least-squares circle fit (Kasa method). Returns (center, radius)."""
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    radius = math.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0))
    return (cx, cy), radius


def simplify_round_polygon_to_octagon(points, min_size=0):
    """
    Replace a near-circular polygon with a regular octagon, using a
    proper least-squares circle fit rather than centroid + mean radius
    for a more accurate center/radius when vertices aren't perfectly
    symmetric.
    """
    (cx, cy), radius = kasa_circle_fit(points)

    if radius * 2 < min_size:
        return np.asarray(points)

    angles = np.deg2rad(np.arange(0, 360, 45) + 22.5)  # 0,45,90,.. +22.5
    octagon = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
    ])
    return octagon


# ============= periphery ring detection =============

def compute_bbox(points_list):
    """Bounding box, in the same ((xmin,ymin),(xmax,ymax)) format gdspy uses."""
    xs = []
    ys = []
    for pts in points_list:
        pts = np.asarray(pts)
        if pts.size == 0:
            continue
        xs.append(pts[:, 0])
        ys.append(pts[:, 1])
    if not xs:
        return None
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    return (float(xs.min()), float(ys.min())), (float(xs.max()), float(ys.max()))


def flatten_subtree_polygons(cell, seen=None):
    """
    Recursively collect all polygon point arrays reachable from `cell`
    through CellReference/CellArray (with their transforms applied),
    without mutating the cell or library.
    """
    if seen is None:
        seen = set()
    if id(cell) in seen:
        return []
    seen.add(id(cell))

    points_list = []
    for poly in cell.polygons:
        for pts in poly.polygons:
            points_list.append(pts)

    for ref in cell.references:
        ref_cell = ref.ref_cell
        if isinstance(ref_cell, str):
            continue  # unresolved reference, nothing to flatten
        for pts in ref.get_polygons():
            points_list.append(pts)

    return points_list


def is_ring_candidate(cell_bbox, design_bbox, extent_fraction_threshold=0.5,
                       dominant_span_threshold=0.85, edge_touch_tolerance=None):
    """
    'On the periphery' test: the candidate's bounding box spans a large
    fraction of the overall design extent in at least one dimension AND
    is close to / touching the design's own bbox edges - i.e. peripheral,
    not just large and centered.

    The design's own overall bbox can be skewed by unrelated peripheral
    elements that extend past the ring itself (e.g. RF/probe pads sticking
    out beyond a seal ring), which would otherwise make a genuine ring's
    bbox appear not to "touch" the design bbox edges within a tight
    tolerance. To stay general without hardcoding a pad/via allowance, a
    candidate whose bbox already spans a dominant fraction of the design
    extent (dominant_span_threshold) in both dimensions is accepted
    without the stricter edge-touch check - nothing in a real layout that
    isn't itself an enclosing/peripheral structure spans nearly the whole
    design in both directions.
    """
    (dxmin, dymin), (dxmax, dymax) = design_bbox
    (cxmin, cymin), (cxmax, cymax) = cell_bbox
    design_w = dxmax - dxmin
    design_h = dymax - dymin
    if design_w <= 0 or design_h <= 0:
        return False

    span_w = (cxmax - cxmin) / design_w
    span_h = (cymax - cymin) / design_h

    if span_w >= dominant_span_threshold and span_h >= dominant_span_threshold:
        return True

    spans_enough = span_w >= extent_fraction_threshold or span_h >= extent_fraction_threshold
    if not spans_enough:
        return False

    tol = edge_touch_tolerance if edge_touch_tolerance is not None else 0.05 * max(design_w, design_h)
    touches_edge = (
        abs(cxmin - dxmin) <= tol or abs(cxmax - dxmax) <= tol or
        abs(cymin - dymin) <= tol or abs(cymax - dymax) <= tol
    )
    return touches_edge


def hollow_core_fraction(points_list, bbox, core_inset_fraction=0.15,
                          max_single_polygon_area_fraction=0.5):
    """
    Union all polygons in `points_list`, inset the candidate's own bbox by
    core_inset_fraction * min(width, height) on all sides, and measure what
    fraction of that inset 'core' rectangle is covered by the union.
    Near 0 => hollow middle => frame/ring shape.

    A single polygon whose own area covers more than
    max_single_polygon_area_fraction of the candidate's bbox area is
    excluded from the union before measuring hollowness. A real ring/frame
    is by definition made of thin segments; a lone polygon that big is a
    non-physical outline/boundary/label marker (common in hierarchical
    EDA exports) rather than drawn fill, and letting it dominate the
    coverage measurement would mask genuinely hollow ring geometry
    underneath it.
    """
    (xmin, ymin), (xmax, ymax) = bbox
    bbox_area = (xmax - xmin) * (ymax - ymin)

    polys = [ShapelyPolygon(p) for p in points_list if len(p) >= 3]
    polys = [p for p in polys if p.is_valid and p.area > 0]
    if not polys:
        return 0.0
    if bbox_area > 0:
        # only drop oversized polygons if other material remains to test
        # against - otherwise a lone solid rectangle (trivially "oversized"
        # relative to its own bbox) would look artificially hollow once
        # excluded, instead of correctly registering as solid
        filtered = [p for p in polys if p.area / bbox_area <= max_single_polygon_area_fraction]
        if filtered:
            polys = filtered
    union = unary_union(polys)

    inset = core_inset_fraction * min(xmax - xmin, ymax - ymin)
    core = ShapelyPolygon([
        (xmin + inset, ymin + inset), (xmax - inset, ymin + inset),
        (xmax - inset, ymax - inset), (xmin + inset, ymax - inset),
    ])
    if core.area <= 0:
        return 0.0
    covered = union.intersection(core).area
    return covered / core.area


def detect_periphery_ring_cells(library, top_cell_name,
                                 extent_fraction_threshold=0.5,
                                 core_inset_fraction=0.15,
                                 hollow_threshold=0.05):
    """
    Walk every cell in the library (except the top cell) and flag cells
    whose flattened subtree geometry forms a hollow frame near the overall
    design's periphery. Purely geometric - does not rely on cell naming.
    Read-only: does not modify the library.

    Returns a list of dicts: {"cell_name", "bbox", "hollow_fraction"}.
    """
    top = library.cells[top_cell_name]
    design_bbox = top.get_bounding_box()
    if design_bbox is None:
        return []

    candidates = []
    for cell in library.cells.values():
        if cell.name == top_cell_name:
            continue
        points_list = flatten_subtree_polygons(cell)
        if not points_list:
            continue
        cell_bbox = compute_bbox(points_list)
        if cell_bbox is None:
            continue
        if not is_ring_candidate(cell_bbox, design_bbox, extent_fraction_threshold):
            continue
        hollow_frac = hollow_core_fraction(points_list, cell_bbox, core_inset_fraction)
        if hollow_frac <= hollow_threshold:
            candidates.append({
                "cell_name": cell.name,
                "bbox": cell_bbox,
                "hollow_fraction": hollow_frac,
            })
    return candidates


def collect_referenced_cell_names(cell, seen=None):
    """
    Recursively collect the names of every distinct cell reachable from
    `cell` through CellReference/CellArray (not including `cell` itself).
    """
    if seen is None:
        seen = set()
    names = []
    for ref in cell.references:
        ref_cell = ref.ref_cell
        if isinstance(ref_cell, str) or ref_cell.name in seen:
            continue
        seen.add(ref_cell.name)
        names.append(ref_cell.name)
        names.extend(collect_referenced_cell_names(ref_cell, seen))
    return names


def delete_polygons_in_cells(library, cell_names):
    """
    Empty the polygon list of each named cell (removing geometry actually
    held there). CellReference objects elsewhere are left untouched, so
    they resolve cleanly to an empty cell rather than dangling.
    Returns {cell_name: number_of_polygons_removed}.
    """
    removed_counts = {}
    for name in cell_names:
        cell = library.cells.get(name)
        if cell is None:
            continue
        removed_counts[name] = len(cell.polygons)
        cell.polygons = []
    return removed_counts


def detect_and_delete_periphery_rings(library, top_cell_name, apply=True,
                                       extent_fraction_threshold=0.5,
                                       core_inset_fraction=0.15,
                                       hollow_threshold=0.05):
    """
    Find thin ring/frame cells near the design periphery and, unless
    apply=False, delete their geometry (deletion is the default). Always
    prints a report of what was found/done.
    """
    candidates = detect_periphery_ring_cells(
        library, top_cell_name,
        extent_fraction_threshold=extent_fraction_threshold,
        core_inset_fraction=core_inset_fraction,
        hollow_threshold=hollow_threshold,
    )

    if not candidates:
        print("Periphery ring detection: no candidate cells found")
        return candidates

    mode = "applying deletion" if apply else "dry run, no changes made"
    print(f"Periphery ring detection: {len(candidates)} candidate cell(s) found ({mode})")
    for candidate in candidates:
        print(f"  cell {candidate['cell_name']}: bbox={candidate['bbox']} "
              f"hollow_fraction={candidate['hollow_fraction']:.3f}")

    if apply:
        for candidate in candidates:
            cell = library.cells.get(candidate["cell_name"])
            if cell is None:
                continue
            # A ring candidate is often a wrapper cell that only holds
            # CellReferences into segment cells (corners/sides) that carry
            # the actual geometry - emptying the wrapper's own (usually
            # empty or marker-only) polygon list wouldn't remove the ring.
            # Target the referenced cells that actually hold geometry
            # instead, and leave the wrapper (and its references) in place.
            # If there are no references, the candidate itself is a leaf
            # cell holding the ring geometry directly.
            referenced_names = collect_referenced_cell_names(cell)
            target_names = referenced_names if referenced_names else [candidate["cell_name"]]
            removed = delete_polygons_in_cells(library, target_names)
            for name, count in removed.items():
                if count:
                    print(f"  Deleted {count} polygon(s) from cell {name} (referenced from {candidate['cell_name']})")

    return candidates
