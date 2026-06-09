# Compatibility Export Loss Report Rubric

- `split_export`: exporter writes Harbor/Pier split layout for supported fields.
- `loss_report`: degraded exports include losses, selected paths, file hashes,
  and ignored aliases.
- `foreign_preservation`: importer keeps foreign extension data under
  compatibility metadata with warnings.
- `roundtrip_conformance`: split-to-native-to-split checks prove canonical
  config, prompt, environment, solution, and tests equality for supported
  Harbor-compatible fields.
- `alias_drift`: structural checks reject mixed native/legacy drift by default.
- `docs`: task authoring docs explain compatibility exports without deprecating
  split layouts.
