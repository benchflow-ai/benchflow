# Terminal-Bench v1 fixtures

`input/` holds four small Terminal-Bench v1 tasks written for these tests (not copied from the benchmark): `hello` has the default compose file, `settings` moves compose settings into the Dockerfile, `services` keeps a multi-service compose file, and `yaml-solution` uses `solution.yaml`, which cannot be converted.

`expected/` was written by Harbor's own converter at `harbor-framework/harbor@09f5b96`:

```sh
harbor task migrate -i input -o expected
```

`yaml-solution` is missing from `expected/` because Harbor refuses it too. `tests/tbench_golden.rs` imports each input with `bench import`'s converter, exports the result back to the split layout, and requires the exported folder to match `expected/` file for file, byte for byte, with the same executable bits.
