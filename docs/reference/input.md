---
title: Input Files
parent: Reference
nav_order: 1
---

# Input Files

## Top-level input

```json
{
  "name": "TestSim",
  "mode": "Mode.json",
  "material": "Material.json",
  "beam": "Beam.json",
  "path": "Path.txt",
  "domain": "Domain.json",
  "settings": "Settings.json"
}
```

`mode`, `material`, `beam`, `path`, and `domain` are required. `name` defaults
to `TestSim`; `settings` is optional. Mode, material, beam, domain, and settings
may be inline JSON objects instead of filenames. Relative filenames are opened
relative to the process working directory.

`beam` and `path` also accept arrays. A filename with one `*` expands to a
numbered sequence beginning at 1 and ending at the first missing file, such as
`Beam*.json` -> `Beam1.json`, `Beam2.json`, .... Beam and path counts must match.

## Material

```json
{
  "constants": {
    "T_init": 300.0,
    "T_liq": 1610.0,
    "k": 30.6,
    "c": 600.0,
    "p": 8000.0
  }
}
```

| Key | Meaning | Units | Required/default |
|:--|:--|:--|:--|
| `k` | thermal conductivity | W m⁻¹ K⁻¹ | required |
| `c` | specific heat | J kg⁻¹ K⁻¹ | required |
| `p` | density | kg m⁻³ | required |
| `T_init` | initial temperature | K | 1273 |
| `T_liq` | liquidus temperature | K | 1610 |

## Beam

```json
{
  "shape": {
    "width_x": 80e-6,
    "width_y": 80e-6,
    "width_z": 10e-6
  },
  "intensity": { "power": 370.0, "efficiency": 0.40 }
}
```

The parser requires `width_x`, `width_y`, `power`, and `efficiency`. `width_z`
defaults to zero if omitted, but a positive value must be supplied for a valid,
nonsingular 3-D kernel. Widths are the $\sigma_i$ parameters in the documented
Gaussian kernel [m], not diameters. `power` is nominal $Q$ [W], and
`efficiency` is the absorbed fraction $\eta$. Runtime power becomes $2\eta Q$
for the half-space image convention.

## Path

Each whitespace-separated data row is

```text
Mode  X(mm)  Y(mm)  Z(mm)  Pmod  tParam
```

A nonnumeric header is ignored. Coordinates are converted from mm to m.

| Field | Line (`Mode=0`) | Spot (`Mode=1`) |
|:--|:--|:--|
| position | segment endpoint | fixed spot location |
| `Pmod` | power multiplier | power multiplier |
| `tParam` | speed [m/s] | dwell [s] |

Condor inserts an initial point at `(0,0,0,t=0)`. A zero or negative line speed
gives that segment zero duration and should be avoided.

## Domain and image boundaries

```json
{
  "domain": {
    "resolution": 25e-6,
    "x": [-0.001, 0.006],
    "y": [-0.001, 0.002],
    "z": [-0.001, 0.0]
  },
  "boundary_conditions": {
    "x": [-0.001, 0.006],
    "y": [-0.001, 0.002],
    "z_min": -0.001,
    "reflections": 1
  }
}
```

`domain.resolution` is required and used on all axes. Bounds are metres. If
$x/y$ bounds are omitted, Condor uses the active path extrema plus a fixed
500 µm margin on each side. Set `z` explicitly: the current unset-z code depends
on the x/y bound state and does not provide one dependable default. Grid counts
are rounded to the nearest interval and bounds are then made grid-consistent.
Boundary-condition keys are optional adiabatic image planes; `reflections`
defaults to 1.

## Global settings

```json
{
  "output": { "dim": "3D", "format": ".csv", "noOutput": false },
  "meltpool": { "radius_check": 250e-6, "trace_radius": 100e-6 },
  "quadrature": { "cutoff_peak": 1e-9, "cutoff_t0tl": 1e-2 },
  "compute": { "prefetch_threads": 4 },
  "mpi": { "print": 1 }
}
```

| Key | Accepted/default | Purpose |
|:--|:--|:--|
| `output.dim` | `2D`, `2.5D`, `3D`, `3D-Full`; `3D` | spatial region written |
| `output.format` | `.csv`, `none`; `.csv` | CSV output or no writer output |
| `output.noOutput` | boolean; `false` | disable file output |
| `meltpool.radius_check` | metres; `0` | lateral perimeter/scan reach. If a scan is outside domain + `meltpool.radius_check` buffer, the timestep is skipped. |
| `meltpool.trace_radius` | metres; `-1` | override automatic beam trace radius for seeding new points for meltpool tracking|
| `quadrature.cutoff_peak` | fraction; `1e-9` | spatial history-radius estimate |
| `quadrature.cutoff_t0tl` | fraction; `1e-2` | spatial history-radius estimate |
| `compute.prefetch_threads` | integer; `2` | host node-preparation workers |
| `mpi.print` | `0`, `1`, or `2`; `0` | mute, rank-zero stdout, or rank logs |

Configuration keys and string values are case-sensitive. Only the values shown
above are accepted.
