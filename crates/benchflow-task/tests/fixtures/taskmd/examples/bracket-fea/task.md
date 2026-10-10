Design an aluminum (6061-T6) wall bracket in CadQuery that holds a 12 kg load hung 150 mm from the wall. The tip may deflect at most 0.5 mm, the bracket must weigh at most 180 g, and it mounts with two M6 screws spaced 60 mm apart vertically. Keep the peak von Mises stress below half the yield strength.

Write `/work/bracket.py` so that running it produces `/work/bracket.step`, with the load, arm length, and screw spacing as parameters. We will re-run your script at other values of those parameters and check each part with our own finite-element analysis.

```toml task
name = "examples/bracket-fea"
title = "A light bracket that passes FEA"
version = "3.2.0"
keywords = ["cad", "mechanical", "fea", "3d"]

[about]
difficulty = "medium"
category = "industrial-physical-systems"

[agent]
timeout = "1h"

[sandbox]
image = "ghcr.io/example/cad-fea@sha256:5c1b0e7d"
cpus = 4
memory = "8 GB"

[conventions]
units = "mm, N, g"
material = "6061-T6: E = 68.9 GPa, yield = 276 MPa, density = 2.70 g/cm3"

[verifier]
timeout = "20m"
isolation = "separate"
```
