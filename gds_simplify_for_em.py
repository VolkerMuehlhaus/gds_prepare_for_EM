# Simplification pipeline and cheap complexity estimate for EM simulation, as a
# library API (no GUI): used by setupEM's Tools > Simplify GDS and layout preview,
# and by the gds2palace MCP server.
# - run_simplify: cutout filling, floating fill removal, per-layer merge and via
#   array merging in the same order as gds_prepare_for_EM.py's main(), followed by
#   a final polygon validity check/repair, written to a new GDS file.
# - layers_from_stackup: the metal layers and via layers (with the metal layers
#   above/below each via) that run_simplify needs, from a parsed gds2palace stackup.
# - estimate_polygon_count / estimate_polygon_count_by_layer: polygon count of a
#   cell without flattening the hierarchy, fast even for dense fill/via arrays.
# - ground_plane_holes / fill_ground_plane_holes: repeated openings in ground
#   planes, measured on the merged layer (holes inside tiles and gaps between
#   tiles alike), and filling exactly those.
# Moved from setupEM (simplify_gds.py, gds_hierarchy_scan.py) unchanged.

########################################################################
#
# Copyright 2025-2026 Volker Muehlhaus and IHP PDK Authors
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

import gdstk

from gds_geometry_utils import cell_bounding_box, cells_by_name, design_cell, polygons_by_spec

# conventional GDS layer range for port (and thermal) polygons in gds2palace/setupEM,
# same fallback as setupEM's Preferences > Ports
DEFAULT_PORT_LAYER_RANGE = (201, 299)

# a merged shape is a ground plane if, with its holes filled, it covers at least
# GROUND_AREA_FRACTION of the layout's bounding box and at least GROUND_FILL of its own
# bounding box (solid, not a network of lines)
GROUND_AREA_FRACTION = 0.10
GROUND_FILL = 0.60


def layers_from_stackup(metals_list, port_layer_range=DEFAULT_PORT_LAYER_RANGE):
    """Metal layers and via layers for run_simplify(), from a gds2palace stackup
    (metals_list from stackup_reader.read_substrate()).
    Returns (metal_layers, via_layers): metal_layers is the sorted list of planar
    metal (conductor or sheet) layer numbers; via_layers is a list of
    (via layer, [metal layers above], [metal layers below]). Layers inside
    port_layer_range are left out, they hold port polygons, not metal.
    """
    layer_min, layer_max = port_layer_range
    metal_layers = sorted(
        int(m.layernum) for m in metals_list.getallplanarmetals()
        if not (layer_min <= int(m.layernum) <= layer_max)
    )
    via_layers = [
        (int(m.layernum),
         [int(n.layernum) for n in m.above if n.is_metal],
         [int(n.layernum) for n in m.below if n.is_metal])
        for m in metals_list.metals
        if m.is_via and not (layer_min <= int(m.layernum) <= layer_max)
    ]
    return metal_layers, via_layers


def _top_cell(lib, cellname):
    """The cell called cellname, else the design cell (gds_geometry_utils.design_cell(): the top-level cell with the
    largest bounding box, not KLayout's $$$CONTEXT_INFO$$$), as the command-line tools choose it. Up to 1.7 this was
    the first top-level cell, which depends on the GDS reader's order (gdspy and gdstk differ) and could be
    $$$CONTEXT_INFO$$$: run_simplify then wrote just that cell. Raises IndexError for a file without cells."""
    cell = cells_by_name(lib).get(cellname)
    if cell is not None:
        return cell
    try:
        return design_cell(lib)[0]
    except ValueError:
        return lib.top_level()[0]


def _flatten_cell(lib, cellname):
    """Return (lib, top_cell) with top_cell flattened in place, its paths
    converted to polygons.

    The former subcells of top_cell that nothing references any more are
    removed from the library. Otherwise they stay in the output as extra
    top-level cells, and a reader that takes the first top-level cell (like
    gds2palace's read_gds() without a cellname) can pick one of them instead
    of the design. Other top-level cells of the input and their subcells are
    kept."""
    top_cell = _top_cell(lib, cellname)
    subcells = set(top_cell.dependencies(True))
    top_cell.flatten()
    paths = top_cell.paths
    if paths:
        top_cell.add(*[p for path in paths for p in path.to_polygons()])
        top_cell.remove(*paths)
    # removing an orphan can orphan its own subcells, so repeat until none is left
    orphans = [c for c in lib.top_level() if c in subcells]
    while orphans:
        lib.remove(*orphans)
        orphans = [c for c in lib.top_level() if c in subcells]
    return lib, top_cell


def run_simplify(gds_path, metal_layers, output_path, cellname="",
                  do_cutouts=False, max_hole_area=None,
                  do_floating=False, fill_minsize=1.0, fill_maxsize=None, fill_mincount=20,
                  do_merge=False, via_layers=None, via_merge_size=0,
                  fill_ground_holes=False, ground_hole_mincount=20):
    """Load gds_path, optionally fill small cutouts, remove floating metal
    fill, and/or merge (boolean OR) polygons per layer as a final pass, on
    the given metal_layers, write the result to output_path. Mirrors the
    step order gds_prepare_for_EM.py's own main() uses: cutout removal first
    (works on the hierarchical design), floating-fill removal after that
    (needs a flattened, per-layer-merged cell first so that touching fill
    tiles aren't mistaken for isolated ones - see
    find_isolated_same_size_polygons_by_layer()'s own docstring), per-layer
    merge, then via array merging.

    via_layers: list of (via layer, metal layers above, metal layers below),
    GDS layer numbers. Vias closer than via_merge_size (microns) are merged
    if they connect the same metal shapes above and below, see
    gds_prepare_for_EM.merge_via_array_by_metal_overlap(). Vias without metal
    above or below are kept unchanged. None or via_merge_size <= 0: no via
    merging.

    fill_ground_holes: fill the repeated openings of ground planes, measured
    on the merged layers (holes inside tiles and gaps between tiles), see
    ground_plane_holes(); ground_hole_mincount openings of one size count as
    repeated. Runs after floating-fill removal / per-layer merge, before via
    merging. No size threshold: only the repeated sizes found are filled.
    """
    from gds_prepare_for_EM import (
        remove_cutout_keep_hierarchy,
        merge_polygons_by_layer,
        find_isolated_same_size_polygons_by_layer,
        merge_polygons,
        merge_via_array_by_metal_overlap,
    )
    from gds_geometry_utils import validate_and_repair_polygons

    lib = gdstk.read_gds(gds_path)
    unit, precision = lib.unit, lib.precision
    top_cell = _top_cell(lib, cellname)

    if do_cutouts:
        design_bbox = top_cell.bounding_box()
        lib = remove_cutout_keep_hierarchy(lib, metal_layers, design_bbox=design_bbox,
                                            max_hole_area=max_hole_area)
        top_cell = _top_cell(lib, cellname)

    # floating-fill removal already ends with a flattened, per-layer-merged
    # cell (it needs that itself, to tell a floating fill cluster apart from
    # touching real metal) - so a separate final merge pass is only extra
    # work when floating-fill removal did NOT already run
    already_merged_flat = False

    if do_floating:
        lib, top_cell = _flatten_cell(lib, cellname)
        merged_lib = merge_polygons_by_layer(top_cell, layers_list=metal_layers)
        merged_top = merged_lib.top_level()[0]
        lib = find_isolated_same_size_polygons_by_layer(
            merged_top, metal_layers,
            minsize=fill_minsize, maxsize=fill_maxsize, mincount=fill_mincount)
        already_merged_flat = True

    if do_merge and not already_merged_flat:
        lib, top_cell = _flatten_cell(lib, cellname)
        lib = merge_polygons_by_layer(top_cell, layers_list=metal_layers)

    if fill_ground_holes:
        lib, top_cell = _flatten_cell(lib, cellname)
        lib, _ = fill_ground_plane_holes(top_cell, metal_layers, mincount=ground_hole_mincount)

    if via_layers and via_merge_size > 0:
        lib, top_cell = _flatten_cell(lib, cellname)
        lpp = polygons_by_spec(top_cell)
        print(f'Via array merging (distance {via_merge_size} um):')

        def metal_polygons(layers):
            polygons = [p for (layer, purpose), polys in lpp.items() if layer in layers for p in polys]
            # touching metal pieces are one metal shape for via grouping
            return merge_polygons(polygons) if polygons else []

        for via_layer, above_layers, below_layers in via_layers:
            purposes = [purpose for (layer, purpose) in lpp if layer == via_layer]
            if not purposes:
                continue
            above, below = metal_polygons(above_layers), metal_polygons(below_layers)
            for purpose in purposes:
                vias = lpp[(via_layer, purpose)]
                merged = merge_via_array_by_metal_overlap(
                    vias, above, below, via_merge_size, keep_unmatched=True)
                # filter: one pass over the cell (remove() searches the cell once per polygon)
                top_cell.filter([(via_layer, purpose)], remove=True, paths=False, labels=False)
                top_cell.add(*[gdstk.Polygon(pts, layer=via_layer, datatype=purpose) for pts in merged])
                print(f'  layer {via_layer}:{purpose}: {len(vias)} -> {len(merged)} polygons')

    # Final polygon validity check/repair - always runs, regardless of which
    # operations above were selected. Catches any polygon still geometrically
    # invalid (self-intersecting) no matter the reason - e.g. a cutout
    # deliberately left alone by remove_cutout_keep_hierarchy() for being
    # larger than max_hole_area, or a merge step re-introducing an invalid
    # "bridge" hole encoding while fusing polygons back together. A
    # solid-body mesher like gds2palace cannot handle this even though
    # GDSII itself tolerates it.
    repaired_count, unresolved = validate_and_repair_polygons(lib, metal_layers)
    if repaired_count:
        print(f'Final validity check: repaired {repaired_count} invalid polygon(s)')
    if unresolved:
        print(f'Final validity check: WARNING - {len(unresolved)} polygon(s) could not be '
              f'auto-repaired, still geometrically invalid in the output file:')
        for cell_name, n, layer, reason in unresolved:
            print(f'  {cell_name} polygon #{n} layer {layer}: {reason}')
    if not repaired_count and not unresolved:
        print('Final validity check: no invalid polygons found')

    # in the unit of the input file; max_points: gdstk would otherwise
    # fracture every polygon over 199 vertices on writing, gdspy didn't
    lib.unit, lib.precision = unit, precision
    lib.write_gds(output_path, max_points=8190)


def _merged_parts(polygons):
    """Union of a layer's polygons, as a list of shapely Polygons (with their holes)."""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    parts = []
    for pts in polygons:
        if len(pts) < 3:
            continue
        geom = make_valid(ShapelyPolygon(pts))
        parts.extend(g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and g.area > 0)
    union = unary_union(parts)
    return [g for g in getattr(union, "geoms", [union]) if g.geom_type == "Polygon"]


def _is_ground_plane(part, layout_area):
    from shapely.geometry import Polygon as ShapelyPolygon
    outline = ShapelyPolygon(part.exterior)
    x0, y0, x1, y1 = outline.bounds
    box_area = (x1 - x0) * (y1 - y0)
    return (outline.area >= GROUND_AREA_FRACTION * layout_area and box_area > 0
            and outline.area / box_area >= GROUND_FILL)


def ground_plane_holes(cell, layers_list, mincount=20, size_tol=1e-3):
    """Openings in ground planes, measured on the merged layers of cell (hierarchy flattened).

    Merging first makes holes drawn inside tiles and gaps left between side-by-side tiles
    the same kind of opening. Openings are grouped by bounding-box size (size_tol, layout
    units); a size with at least mincount openings on a layer/datatype is repeated (cheesing,
    a tiled mesh), the others are kept (openings that belong to the circuit). Openings in
    shapes that are not ground planes are only counted.

    Returns {(layer, datatype): {"ground_planes": n, "repeated": {(w, h): [holes]},
    "other": {(w, h): [holes]}, "outside_ground": n}} with shapely hole Polygons.
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    (x0, y0), (x1, y1) = cell_bounding_box(cell)
    layout_area = (x1 - x0) * (y1 - y0)
    result = {}
    for (layer, datatype), polys in polygons_by_spec(cell).items():
        if layer not in layers_list or not polys:
            continue
        parts = _merged_parts(polys)
        grounds = [g for g in parts if _is_ground_plane(g, layout_area)]
        sizes = {}
        for g in grounds:
            for ring in g.interiors:
                hole = ShapelyPolygon(ring)
                hx0, hy0, hx1, hy1 = hole.bounds
                key = (round((hx1 - hx0) / size_tol), round((hy1 - hy0) / size_tol))
                sizes.setdefault(key, []).append(hole)
        outside = sum(len(g.interiors) for g in parts if not _is_ground_plane(g, layout_area))
        if grounds or outside:
            result[(int(layer), int(datatype))] = {
                "ground_planes": len(grounds),
                "repeated": {(w * size_tol, h * size_tol): holes for (w, h), holes in sizes.items()
                             if len(holes) >= mincount},
                "other": {(w * size_tol, h * size_tol): holes for (w, h), holes in sizes.items()
                          if len(holes) < mincount},
                "outside_ground": outside,
            }
    return result


def fill_ground_plane_holes(cell, layers_list, mincount=20, size_tol=1e-3):
    """Fill the repeated openings of the ground planes (see ground_plane_holes()) in a
    flattened cell. Returns (new library with one cell of the same name, measurement)."""
    import numpy as np
    from gds_prepare_for_EM import merge_polygons

    measured = ground_plane_holes(cell, layers_list, mincount, size_tol)
    new_lib = gdstk.Library()
    new_cell = gdstk.Cell(cell.name)
    for (layer, datatype), polys in polygons_by_spec(cell, depth=0).items():
        fill = []
        for (w, h), holes in measured.get((int(layer), int(datatype)), {}).get("repeated", {}).items():
            print(f"Layer {layer}:{datatype}: filled {len(holes)} repeated ground plane holes of size ({w:g},{h:g})")
            fill.extend(np.array(hole.exterior.coords)[:-1] for hole in holes)
        merged = merge_polygons(list(polys) + fill) if fill else polys
        new_cell.add(*[gdstk.Polygon(pts, layer=layer, datatype=datatype) for pts in merged])
    new_lib.add(new_cell)
    return new_lib, measured


def open_gds_cell(gds_path, cellname):
    """Open gds_path with gdstk and resolve the target cell: cellname if
    given and found, else the design cell (_top_cell()). Raises on a bad
    file or a library with no top-level cell - callers already wrap their
    own call sites in a broad try/except.

    Returns gdstk objects (library, cell). Up to version 1.7 these were
    gdspy objects; setupEM and the gds2palace MCP server walk them.
    """
    library = gdstk.read_gds(gds_path)
    if not library.top_level():
        raise ValueError(f"no top-level cell in {gds_path}")
    return library, _top_cell(library, cellname)


def _placements(ref):
    """How many times a reference places its cell (an array reference: columns * rows)."""
    return max(ref.repetition.size, 1)


def estimate_polygon_count(gds_path, cellname, layernumbers, purposelist):
    """Fast pre-check of how many polygons a full read_gds()-based load
    would end up processing, without flattening the GDS hierarchy. Walks
    the cell/reference tree directly: a plain CellReference contributes its
    referenced cell's own (memoized) count once, a CellArray contributes it
    columns*rows times - the same arithmetic flatten() would otherwise do
    by actually generating that many polygon objects. Returns None if the
    file can't even be opened here (the real read_gds() call is expected to
    surface that error properly instead).
    """
    try:
        library, cell = open_gds_cell(gds_path, cellname)
    except Exception:
        return None

    layernumbers = set(int(n) for n in layernumbers)
    purposeset = set(int(p) for p in purposelist) if purposelist else None

    memo = {}

    def count(c):
        if c.name in memo:
            return memo[c.name]
        total = 0
        for polygon in c.polygons:
            if polygon.layer in layernumbers and (purposeset is None or polygon.datatype in purposeset):
                total += 1
        for ref in c.references:
            if not isinstance(ref.cell, gdstk.Cell):
                continue  # cell missing from the file
            total += count(ref.cell) * _placements(ref)
        memo[c.name] = total
        return total

    return count(cell)


def estimate_polygon_count_by_layer(gds_path, cellname, purposelist=None):
    """Like estimate_polygon_count(), but for every layer at once:
    {layer number: polygon count}, without flattening the hierarchy.
    Raises on a file that can't be opened (unlike estimate_polygon_count).
    """
    library, cell = open_gds_cell(gds_path, cellname)
    purposeset = set(int(p) for p in purposelist) if purposelist else None

    memo = {}

    def count(c):
        if c.name in memo:
            return memo[c.name]
        totals = {}
        for polygon in c.polygons:
            if purposeset is None or polygon.datatype in purposeset:
                totals[polygon.layer] = totals.get(polygon.layer, 0) + 1
        for ref in c.references:
            if not isinstance(ref.cell, gdstk.Cell):
                continue  # cell missing from the file
            factor = _placements(ref)
            for layer, n in count(ref.cell).items():
                totals[layer] = totals.get(layer, 0) + n * factor
        memo[c.name] = totals
        return totals

    return dict(sorted(count(cell).items()))
