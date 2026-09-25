---
title: Case Setup
parent: User Guide
nav_order: 6
---

# Case Setup

Condor cases are kept outside the source distribution. A case directory should
contain a top-level `ParamInput.json` and the files it references:

```text
my-case/
├── ParamInput.json
├── Beam.json
├── Domain.json
├── Material.json
├── Mode.json
├── Path.txt
└── Settings.json
```

Run from inside the case so relative paths resolve:

```bash
cd /path/to/my-case
/path/to/Condor/build/apps/Main ParamInput.json
```

Generated `Data/` and calibration products are runtime artifacts; they are not
required inputs for a new case. See [Input Files]({{ '/reference/input/' | relative_url }})
and [Modes and Hooks]({{ '/reference/modes/' | relative_url }}) for complete
configuration details.
