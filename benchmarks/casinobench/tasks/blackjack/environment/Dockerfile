# Task-package contract requires environment/Dockerfile. CasinoBench is
# manifest-backed: when run with --environment-manifest, the manifest's run
# `image` (casinobench-base:latest) is the sandbox target and THIS file is
# bypassed. It is a trivial passthrough so the per-game task carries no per-task
# build (the game is chosen at runtime by CASINOBENCH_GAME, not baked here), and
# so the non-manifest path still gets the working casino-service image.
FROM casinobench-base:latest
