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

import os
import tempfile

import gdspy

# conventional GDS layer range for port (and thermal) polygons in gds2palace/setupEM,
# same fallback as setupEM's Preferences > Ports
DEFAULT_PORT_LAYER_RANGE = (201, 299)


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


def _flatten_cell(lib, cellname):
    """Return (lib, top_cell) with top_cell flattened in place. Round-trips
    through a temporary GDS file first: gdspy's flatten()/get_polygonsets()
    hits an internal bug on some in-memory-only reference structures ('tuple'
    object does not support item assignment) that only a GDS write+read
    cycle avoids - same workaround gds_prepare_for_EM.py's own CLI pipeline
    uses before every flatten() call in its main().

    The former subcells of top_cell that nothing references any more are
    removed from the library. Otherwise they stay in the output as extra
    top-level cells, and a reader that takes the first top-level cell (like
    gds2palace's read_gds() without a cellname) can pick one of them instead
    of the design. Other top-level cells of the input and their subcells are
    kept."""
    with tempfile.TemporaryDirectory(prefix="setupEM_simplify_") as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "pre_flatten.gds")
        lib.write_gds(tmp_path)
        lib = gdspy.GdsLibrary(infile=tmp_path)
    top_cell = lib.cells.get(cellname, lib.top_level()[0])
    subcells = top_cell.get_dependencies(True)
    top_cell.flatten()
    # removing an orphan can orphan its own subcells, so repeat until none is left
    orphans = [c for c in lib.top_level() if c in subcells]
    while orphans:
        for cell in orphans:
            lib.remove(cell)
        orphans = [c for c in lib.top_level() if c in subcells]
    return lib, top_cell


def run_simplify(gds_path, metal_layers, output_path, cellname="",
                  do_cutouts=False, max_hole_area=None,
                  do_floating=False, fill_minsize=1.0, fill_maxsize=None, fill_mincount=20,
                  do_merge=False, via_layers=None, via_merge_size=0):
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
    """
    from gds_prepare_for_EM import (
        remove_cutout_keep_hierarchy,
        merge_polygons_by_layer,
        find_isolated_same_size_polygons_by_layer,
        merge_polygons,
        merge_via_array_by_metal_overlap,
    )
    from gds_geometry_utils import validate_and_repair_polygons

    # merge_polygons_by_layer()/find_isolated_same_size_polygons_by_layer()
    # create cells via the bare gdspy.Cell(...) constructor, which
    # auto-registers each one into gdspy's persistent global
    # gdspy.current_library - harmless for the CLI (a fresh process every
    # run) but in this long-running GUI, a second Run on the same file
    # would reuse the same cell names and gdspy raises "already present in
    # library". Starting from a fresh current_library each call clears that
    # accumulated state without touching gds_prepare_for_EM itself.
    gdspy.current_library = gdspy.GdsLibrary()

    lib = gdspy.GdsLibrary()
    lib.read_gds(gds_path)
    top_cell = lib.cells.get(cellname, lib.top_level()[0])

    if do_cutouts:
        design_bbox = top_cell.get_bounding_box()
        lib = remove_cutout_keep_hierarchy(lib, metal_layers, design_bbox=design_bbox,
                                            max_hole_area=max_hole_area)
        top_cell = lib.cells.get(cellname, lib.top_level()[0])

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

    if via_layers and via_merge_size > 0:
        lib, top_cell = _flatten_cell(lib, cellname)
        lpp = top_cell.get_polygons(by_spec=True)
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
                top_cell.remove_polygons(
                    lambda pts, layer, datatype: layer == via_layer and datatype == purpose)
                for pts in merged:
                    top_cell.add(gdspy.Polygon(pts, layer=via_layer, datatype=purpose))
                print(f'  layer {via_layer}:{purpose}: {len(vias)} -> {len(merged)} polygons')

    # Final polygon validity check/repair - always runs, regardless of which
    # operations above were selected. Catches any polygon still geometrically
    # invalid (self-intersecting) no matter the reason - e.g. a cutout
    # deliberately left alone by remove_cutout_keep_hierarchy() for being
    # larger than max_hole_area, or a merge step re-introducing an invalid
    # "bridge" hole encoding while fusing polygons back together. A
    # solid-body mesher like gds2palace cannot handle this even though
    # gdspy/GDSII themselves tolerate it.
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

    lib.write_gds(output_path)


def open_gds_cell(gds_path, cellname):
    """Open gds_path and resolve the target cell: cellname if given and
    found, else the file's first top-level cell. Shared first step for
    every walker below. Raises on a bad file or a library with no top-level
    cell, same as gdspy itself would - callers already wrap their own call
    sites in a broad try/except.
    """
    library = gdspy.GdsLibrary(infile=gds_path)
    top_level = library.top_level()
    if not top_level:
        raise ValueError(f"no top-level cell in {gds_path}")
    cell = library.cells.get(cellname, top_level[0])
    return library, cell


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
        for polygonset in c.polygons:
            for layer, datatype in zip(polygonset.layers, polygonset.datatypes):
                if int(layer) in layernumbers and (purposeset is None or int(datatype) in purposeset):
                    total += 1
        for ref in c.references:
            ref_cell = ref.ref_cell
            if isinstance(ref_cell, str):
                ref_cell = library.cells.get(ref_cell)
            if ref_cell is None:
                continue
            sub = count(ref_cell)
            total += sub * ref.columns * ref.rows if isinstance(ref, gdspy.CellArray) else sub
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
        for polygonset in c.polygons:
            for layer, datatype in zip(polygonset.layers, polygonset.datatypes):
                if purposeset is None or int(datatype) in purposeset:
                    totals[int(layer)] = totals.get(int(layer), 0) + 1
        for ref in c.references:
            ref_cell = ref.ref_cell
            if isinstance(ref_cell, str):
                ref_cell = library.cells.get(ref_cell)
            if ref_cell is None:
                continue
            factor = ref.columns * ref.rows if isinstance(ref, gdspy.CellArray) else 1
            for layer, n in count(ref_cell).items():
                totals[layer] = totals.get(layer, 0) + n * factor
        memo[c.name] = totals
        return totals

    return dict(sorted(count(cell).items()))
