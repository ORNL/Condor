---
title: Output
parent: Reference
nav_order: 4
---

# Output

CSV output is written to a `Data/` directory beside the top-level input file,
even though referenced input filenames are resolved from the working directory.
With run name `TestSim`, periodic files are `TestSim_<iteration>.csv` and the
final Interface file is `TestSim_Final.csv`.

## Spatial selection

| `output.dim` | Rows written |
|:--|:--|
| `2D` | top surface of columns that have melted |
| `2.5D` | top plus deepest melted point, identified by `surface` |
| `3D` | melted depth of columns that have melted |
| `3D-Full` | every point in the domain |

The values are case-sensitive. Dense Snapshots mode (`tracking: "None"`)
forces `3D-Full` output.

Coordinates are metres. Other columns depend on enabled mode outputs:

| Column | Meaning | Units |
|:--|:--|:--|
| `T` | temperature | K |
| `tSol` | refined liquidus-crossing time | s |
| `G` | gradient magnitude | K/m |
| `Gx`, `Gy`, `Gz` | gradient components | K/m |
| `G_unit_*` | normalized gradient direction | 1 |
| `V` | isotherm speed | m/s |
| `dTdt` | cooling-rate magnitude | K/s |
| `numMelt` | number of solid-to-liquid transitions | 1 |
| `MP_Length`, `MP_Width`, `MP_Depth` | maximum mapped melt-pool dimensions | m |
| `Col_Depth` | interpolated liquid depth | m |
| `eqFrac` | equiaxed fraction estimate | 1 |

For standalone `Interface` output, solidification quantities describe the
point's current state. When a point melts, `tSol`, `G`, `V`, `dTdt`, and enabled
gradient-vector fields are reset to zero at the next host transfer. If the point
solidifies again first, the pending molten marker is replaced by the newer
solidification event. `numMelt` is preserved and increments once for every
solid-to-liquid transition.

## Melt-pool statistics

When time statistics are enabled, `<name>_MP_Stats.csv` contains
`t,iter,MP_Count,MP_Length,MP_Width` and, for a volumetric domain,
`MP_Depth`. Each row reports the largest detected component dimensions for that
time. Geometry is interpolated at the liquidus between grid points.

`output.noOutput: true`, output format `none`, and coupled RDF/SRDF mode
suppress ordinary writer files.
