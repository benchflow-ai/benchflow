Write and run a protocol for the liquid handler that makes a 1:2 serial dilution of the dye stock across row A of a 96-well plate: eight wells, 200 µL final volume in each, starting from the undiluted stock in A1. Then read absorbance at 630 nm and save the readings with your plate map to `/work/report.csv`.

The deck layout, labware, and stock concentration are in `/lab/deck.json`.

```toml task
name = "examples/serial-dilution"
title = "Serial dilution on a liquid handler"
version = "2.3.0"
keywords = ["wet-lab", "liquid-handling", "protocol"]

[agent]
timeout = "30m"

[sandbox]
image = "ghcr.io/example/opentrons-dev:7.3"

[world]
kind = "simulated"
profile = "opentrons/ot2@7.3"
interface = "protocol-file"
deliverable = "/work/protocol.py"

[world.lab]
deck = "world/deck.json"
api_level = "2.20"
dry_run = "opentrons_simulate"
biosafety_level = 1

[[world.lab.reagent]]
name = "dye stock"
concentration = "100 uM"
volume = "5 mL"

[[world.lab.labware]]
slot = 1
type = "corning_96_wellplate_360ul_flat"

[[world.lab.labware]]
slot = 2
type = "nest_12_reservoir_15ml"

[[world.lab.labware]]
slot = 3
type = "opentrons_96_tiprack_300ul"

[[world.safety.limit]]
quantity = "aspirate_volume"
max = "300 uL"
enforced_by = "dry-run"
measured_by = "protocol-analysis"
on_violation = "reject-protocol"

[world.record]
streams = ["protocol-run-log"]
required = ["protocol-run-log"]

[tiers.real]
kind = "physical"
profile = "opentrons/ot2-bench-3@7.3"
gate = "the simulated tier passes"
supervision = "operator-present"
record = ["run-log", "plate-reader-export", "deck-video"]
```
