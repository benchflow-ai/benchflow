You need to calculate the mass of a 3D-printed part. The input (`/root/scan_data.stl`) is a binary STL, but the 2-byte "Attribute Byte Count" at the end of each triangle record stores the **Material ID** of the object.

1. Parse the binary STL and identify the **largest connected component**, filtering out scanning debris.
2. Look up the Material ID in `/root/material_density_table.md` to find its density.
3. Compute `Volume * Density` and save the result to `/root/mass_report.json`:

```json
{"main_part_mass": 12345.67, "material_id": 42}
```

The result counts as correct within **0.1%**.

```toml task
name = "skillsbench/3d-scan-calc"
title = "Mass of a 3D-printed part from a tagged STL scan"
version = "1.1.0"
authors = ["Wengao Ye"]
keywords = ["3d-geometry", "stl", "binary-parsing"]

[about]
difficulty = "hard"
category = "industrial-physical-systems"
source = "https://github.com/benchflow-ai/skillsbench"

[agent]
timeout = "15m"

[sandbox]
cpus = 1
memory = "4 GB"
disk = "10 GB"
build_timeout = "10m"

[verifier]
timeout = "15m"
```
