# Prepare IHP SG13G2 GDSII layout for EM simulation

This is a collection of tools to simplify GDSII layout for EM simulation.

Layouts are usually simple and clean in the initial design phase, but simulating  a „final“ layout that was already prepared for tape-out with density rules etc. can be a challenge.
The picture below shows some typical challenges:

[<img src="./doc/png/inital_gds.png" width="800" />](initial gds)

1) To fulfill metal density rules, larger areas have been created as an array of squares with hole inside. This hole does not really matter for EM results, but it will lead to additional mesh cells and slow down mesh generation and simulation.
For openEMS, the value of refined_cellsize can efficiently be used to skip small detail, but still those edges will slow down the meshing processing. For Palace, such small detail will all be included in mesh and it is absolutely required to remove this.
2) On layer TopMetal2 shown as orange boxes on the top right side, and many other layers hidden here, the layout includes unconnected (floating) metal boxes that are used to fulfill density rules. Unlike auto-generated dummy metal fill, this man-made dummy metal fill is on purpose „drawing“ and can’t be skipped by the purpose (data type).
3) Especially for pads, there is a massive amount of vias located in via arrays at rather large spacing. We can’t simplify increase the distance for via array merging in the gds2palace or gds2openEMS scripts, because that is a global setting and might also create unintentional short between adjacent via stacks.
4) In the case shown here, the pads for copper pillar are round, which is represented in GDSII as a polygon with many vertices, resulting in over-meshing at these polygons, wasting simulation time.

To solve these issues and create a more simulation-friendly layout, a collection of tools is provided here.

## gds_removefill
Removes unconnected (floating) man-made dummy metal fill. A shape only counts as removable fill once that same size repeats often enough on a layer, so genuinely isolated design content is left alone.

Options: `--mincount N` how many repeats of the same size count as fill (default 20) · `--minsize`/`--maxsize` optional manual size bounds (default: no upper limit) · optional output filename as second argument.

```
python gds_removefill.py layout.gds cleaned_layout.gds --minsize 1 --maxsize 40 --mincount 20
```

## gds_simplify
Replaces density-fill cutouts (squares/shapes with a hole punched in them) with solid outlines, replaces circle-like pads with octagons for faster meshing, and detects and removes thin ring/frame structures on the layout periphery such as seal rings. Always processes the full built-in default set of layers.

Options: `--exclude-layers L1,L2,...` skip specific layers from that default set.

```
python gds_simplify.py layout.gds --exclude-layers 126,134
```

## gds_prepare_for_EM
The all-in-one tool: runs all of the above plus via-array simplification (replacing dense via arrays with a handful of clean shapes) and round-pad-to-octagon conversion, producing a single simulation-ready output file in one pass.

Options: `--fill-mincount`/`--fill-minsize`/`--fill-maxsize` same meaning as in gds_removefill · optional output filename as second argument.

```
python gds_prepare_for_EM.py layout.gds cleaned_layout.gds --fill-minsize 1 --fill-maxsize 40 --fill-mincount 20
```

Starting from the example above, the resulting cleaned and simplified GDSII then looks like this:

[<img src="./doc/png/cleaned_gds.png" width="800" />](cleaned gds)


# Usage
To run a tool, specify the *.gds filename as commandline parameter; the cleaned file is saved with an appropriate file suffix (or pass a second argument to choose the output filename). Every tool always prints its full list of options and the values used for that particular run, so you can see exactly what a run did after the fact - run with `--help` to see the options without processing a file.

example:
```
python gds_prepare_for_EM.py layout.gds
python gds_prepare_for_EM.py layout.gds cleaned_layout.gds
```

# Prerequisites
The code requires Python3 and these libraries, install using pip install libraryname (or `pip install -r requirements.txt`)
- gdspy
- rtree
- numpy
- shapely (>=2.0)



