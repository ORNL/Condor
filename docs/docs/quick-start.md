---
title: Quick Start
parent: User Guide
nav_order: 2
---

# Quick Start

After [building Condor](../installation/), clone the example inputs into a
separate directory and run the Classic case:

```bash
git clone --branch example-inputs --single-branch \
  https://github.com/ORNL/condor.git condor-example-inputs
cd condor-example-inputs/examples/Classic
/path/to/Condor/build/apps/Main ParamInput.json
```

The example inputs are maintained on the repository's
[`example-inputs` branch](https://github.com/ORNL/condor/tree/example-inputs).
Condor resolves input paths against the process working directory, so run the
application from the case directory. Results appear in `Data/`.

The top-level file selects the other inputs:

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

With `Interface` mode, Condor advances a fixed timestep, follows the
liquid/solid interface, and stops once all paths have ended and no liquid points
remain.

Run with MPI in the same working directory:

```bash
mpirun -np 4 /path/to/Condor/build/apps/Main ParamInput.json
```

If no command-line argument is supplied, `Main` opens `ParamInput.json` in the
current directory.

## A first modification

Start from a complete input set, then change one item at a time:

1. Set `intensity.power` and `intensity.efficiency` in `Beam.json`.
2. Set the material constants and liquidus temperature in `Material.json`.
3. Set grid bounds and `resolution` in `Domain.json`.
4. Choose an `Interface.settings.timestep` small enough to resolve motion of the
   melt pool across the grid.
5. Ensure `meltpool.radius_check` exceeds the expected lateral melt-pool reach.

See [Input Files]({{ '/reference/input/' | relative_url }}) for units and accepted forms.
