# Extract objects on *all* IHP SG13G2 layers in GDSII file
# Find polygons with holes (cutouts) that are used to increase density, replace by solid outline
# Find circles and replace by octagon
# Find thin ring/frame structures on the layout periphery (e.g. seal rings) and delete them

# Usage: gds_simplify <input.gds> [--layers L1,L2,...] [--apply-ring-deletion]


# File history:
# Initial version 29 June 2025 Volker Muehlhaus
# Circle detection 01 Dec 2025 Volker Muehlhaus
# Generalized hole/circle detection (shapely) + periphery ring deletion 16 Aug 2026 Volker Muehlhaus

########################################################################
#
# Copyright 2025 Volker Muehlhaus and IHP PDK Authors
#
# Licensed under the GNU General Public License, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/gpl-3.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
########################################################################


import argparse

import gdspy

from gds_geometry_utils import (
    decompose_polygon_holes,
    is_circle_like,
    simplify_round_polygon_to_octagon,
    is_ring_candidate,
    detect_and_delete_periphery_rings,
)


# ==================== settings =========================


# ============= technology specific stuff ===============

# Layer number <-> name mapping. To adapt this tool to a different
# PDK/layer stack, edit this table - metal_layers_list below derives from
# it automatically.
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
}
NAME_TO_LAYER = {name: layer for layer, name in LAYER_NAMES.items()}

# list of layers to evaluate, we only check  metal layers (this combines
# metal and via layers, unlike gds_prepare_for_EM.py's metal_layers_list -
# and unlike that one, this list omits the MIM layer)
metal_layers_list = [layer for name, layer in NAME_TO_LAYER.items() if name != "MIM"]


# list of purpose to evaluate
required_purpose_list = [
  0,  # drawing
  35, # pillar
  70  # copper
]

# list of layers to delete, to reduce file size
delete_layers_list = [
  148,
  160
]

# list of purpose to delete, with no replacement
delete_purpose_list = [
  22,
  23,
  29,
  2 # pin
]

# a polygon-with-hole where the hole covers this much of the exterior area
# is treated as a thin ring, not "fill with cutout" - it gets deleted
# instead of replaced by a solid shape (see detect_and_delete_periphery_rings
# for the hierarchical/multi-polygon version of ring detection)
RING_HOLE_FRACTION_THRESHOLD = 0.90


def print_run_config(parser, args):
  """Print the full set of available commandline options and how to use
  them, followed by the value actually used for each on this run - so a
  user never has to guess what a run did after the fact."""
  print(parser.format_help())
  print("Resolved configuration for this run:")
  for name, value in sorted(vars(args).items()):
    print(f"  {name} = {value}")
  print()


# ============= main ===============

def main():
  parser = argparse.ArgumentParser(description="Simplify IHP SG13G2 GDSII layout for EM simulation.")
  parser.add_argument("input_gds", help="input GDSII file")
  parser.add_argument("--exclude-layers", help="comma-separated layer numbers to exclude from the built-in default set")
  args = parser.parse_args()
  print_run_config(parser, args)

  input_name = args.input_gds
  exclude_layers = set(int(x) for x in args.exclude_layers.split(",")) if args.exclude_layers else set()
  layers_list = [l for l in metal_layers_list if l not in exclude_layers]

  print ("Input file: ", input_name)
  # get basename of input file, append suffix to identify output polygons
  output_name = input_name.replace(".gds","_forEM.gds")

  # Read GDSII library
  output_library = gdspy.GdsLibrary(infile=input_name)

  top_cells = output_library.top_level()
  top_cell_name = top_cells[0].name if top_cells else None
  design_bbox = top_cells[0].get_bounding_box() if top_cells else None

  # iterate over cells
  for cell in output_library:
    print('cellname = ' + str(cell.name))

    # iterate over polygons
    for n,poly in enumerate(cell.polygons):
      # points of this polygon
      polypoints = poly.polygons[0]

      poly_layer = poly.layers[0]
      poly_purpose = poly.datatypes[0]


      # -------- Check for polygons with holes (dummy fill with cutout, or thin rings) ---------
      if ((poly_layer in layers_list or poly_layer>200) and (poly_purpose in required_purpose_list)):
        # Polygon of interest, check if we need to process this

        decomp = decompose_polygon_holes(polypoints)

        if decomp is not None:
          is_thin_ring = (
            decomp["hole_area_fraction"] >= RING_HOLE_FRACTION_THRESHOLD
            and design_bbox is not None
            and is_ring_candidate(poly.get_bounding_box(), design_bbox)
          )

          if is_thin_ring:
            print('   Deleting periphery ring polygon #', str(n), ' layer ', str(poly_layer))
            poly.layers=[0]
            cell.remove_polygons(lambda pts, layer, datatype:layer == 0)
          else:
            print('   Replacing cutout polygon #', str(n), ' layer ', str(poly_layer))
            # invalidate original polygon
            poly.layers=[0]
            # remove original polygon
            cell.remove_polygons(lambda pts, layer, datatype:layer == 0)

            # Replace it with a solid version of the exterior outline
            basepoly = gdspy.Polygon(decomp["exterior_coords"], layer=poly_layer, datatype=poly_purpose)
            cell.add(basepoly)

        # now do the check for circle-like structures (only if not already handled as a hole/ring above)
        elif len(polypoints) > 11 and is_circle_like(polypoints):
          new_points = simplify_round_polygon_to_octagon(polypoints)

          # invalidate original polygon
          poly.layers=[0]
          # remove original polygon
          cell.remove_polygons(lambda pts, layer, datatype:layer == 0)
          basepoly = gdspy.Polygon(new_points, layer=poly_layer, datatype=poly_purpose)
          cell.add(basepoly)


      # -------- Check for layer or purpose on delete list ---------
      if ((poly_layer in delete_layers_list) or (poly_purpose in delete_purpose_list)):
        # invalidate original polygon
        # mark by assigning layer 0
        poly.layers=[0]
        # remove original polygon that is now on layer 0
        cell.remove_polygons(lambda pts, layer, datatype:layer == 0)

  # -------- Detect and delete thin ring/frame structures on the periphery ---------
  # This runs at the hierarchy level (a seal ring is typically built from several
  # segment cells, not one polygon-with-hole), so it runs once per design, after
  # the per-polygon loop above.
  if top_cell_name is not None:
    print()
    detect_and_delete_periphery_rings(output_library, top_cell_name, apply=True)

  # write to output file
  output_library.write_gds(output_name)

  print('\n\nFINISHED: Created output file ', output_name)


if __name__ == "__main__":
  main()
