---
title: Modes and Hooks
parent: Reference
nav_order: 2
---

# Modes and Hooks

A mode file must contain exactly one of `Interface`, `Snapshots`, or
`Calibrate`.

## Interface

```json
{
  "Interface": {
    "settings": { "timestep": 1e-4 },
    "output": { "frequency": 10, "temperature": false },
    "hooks": {
      "solidification": {
        "params": { "dT_err": 1e-3, "max_iter": 10 },
        "outputs": {
          "tSol": true, "G": true, "G_vector": true,
          "G_unit_vector": false, "V": true,
          "dTdt": true, "numMelt": true
        }
      },
      "meltpool": {
        "params": { "MP_Stats_Output": "SpaceAndTime" },
        "outputs": { "MP_Stats": true, "Col_Depth": true }
      },
      "cet": {
        "params": { "N0": 3e13, "n": 3, "a": 1.25e6 },
        "outputs": { "eqFrac": true }
      }
    }
  }
}
```

`settings.timestep` is required and is in seconds. `output` is required;
`frequency` defaults to 0 and `temperature` to false. Frequency 0 suppresses
periodic files but the final file path is still reached. The loop continues
past the last scan segment until the tracked liquid region disappears.

Hook objects are optional. An empty or missing hook is disabled.

- `solidification`: controls liquidus crossing refinement and fields described
  in [Solidification Quantities]({{ '/model/solidification/' | relative_url }}).
  These fields are cleared when a tracked point becomes liquid and are replaced
  when it solidifies again. `numMelt` counts solid-to-liquid transitions.
- `meltpool`: `MP_Stats` requests length, width, and (for a 3D domain) depth.
  `MP_Stats_Output` may be `Space`, `Time`, or `SpaceAndTime`. `Col_Depth`
  records interpolated depth.
- `cet`: requires `N0`, `n`, and `a`; `eqFrac` is a required boolean when the
  hook is configured.

Coupled RDF/SRDF calls force Interface file output and JSON hooks off, then
enable the calculations required by the selected Stork container.

## Snapshots

```json
{
  "Snapshots": {
    "settings": {
      "tracking": "Interface",
      "times": ["10%", "50%", "100%"]
    },
    "output": { "temperature": true },
    "hooks": {
      "meltpool": {
        "outputs": { "MP_Stats": true, "Col_Depth": true }
      }
    }
  }
}
```

`settings.times` must contain at least one finite, nonnegative time and must be
strictly increasing after percentage strings are expanded relative to the
longest scan duration. `tracking` is `None` (default) or `Interface`.

- `None` evaluates every grid point and forces full-volume 3D output.
- `Interface` traces the melted region, reducing work and output volume.

The optional snapshot melt-pool hook records one statistics row per snapshot
and/or maps column depth to snapshot output.

## Calibrate

Calibrate searches multiplicative parameters $\alpha_c$ and $\beta_c$ around
the input material. A candidate changes diffusivity and volumetric heat
capacity as

$$
\alpha'=\alpha_0\alpha_c,\qquad
(\rho c_p)'=\frac{(\rho c_p)_0}{\beta_c}.
$$

The output material keeps input density, uses
$c_p'=(\rho c_p)'/\rho$, and sets $k'=\alpha'(\rho c_p)'$.

### Line evaluator with grid search

```json
{
  "Calibrate": {
    "evaluator": {
      "type": { "line": {
        "velocity": 0.8, "length": 0.0006,
        "target_length": 240e-6,
        "target_width": 160e-6, "target_depth": 55e-6
      }},
      "weights": { "length": 1, "width": 1, "depth": 1, "distance": 0 },
      "loss": "percent"
    },
    "optimizer": {
      "type": { "grid": {
        "points": 5, "iterations": 6,
        "initial_span": 1, "tolerance": 1e-6
      }},
      "output": {
        "file": "Material-Optimized.json",
        "history": "Calibration-History.csv",
        "print": "on"
      }
    }
  }
}
```

### Path evaluator with gradient descent

```json
{
  "Calibrate": {
    "evaluator": {
      "type": { "path": {
        "time": [0.0005, 0.0010, 0.0015],
        "weights": [1, 1, 2],
        "target_length": [180e-6, 230e-6, 260e-6],
        "target_width": [110e-6, 145e-6, 165e-6],
        "target_depth": [35e-6, 50e-6, 60e-6]
      }},
      "weights": { "length": 1, "width": 1, "depth": 1, "distance": 0.05 },
      "loss": "percent"
    },
    "optimizer": {
      "type": { "gradient_descent": {
        "finite_difference": 0.01,
        "learning_rate": 0.25,
        "max_step": 0.25,
        "tolerance": 1e-3,
        "max_iterations": 100
      }},
      "output": {
        "file": "Material-Path-Gradient.json",
        "history": "Calibration-Path-Gradient.csv",
        "print": "on"
      }
    }
  }
}
```

Evaluator type is exactly one of:

- `line`: positive `velocity` and `length`, with one or more of
  `target_length`, `target_width`, `target_depth`.
- `path`: arrays `time`, optional per-time `weights`, and target arrays such as
  `target_width`. It evaluates the configured input path.

Loss is `percent` or `absolute`. Optimizer type is `grid` (odd `points` >= 3)
or `gradient_descent` (`finite_difference`, `learning_rate`, `max_step`,
`tolerance`, and `max_iterations`). Configuration keys and string values are
case-sensitive.
