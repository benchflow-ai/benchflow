# Task-package contract requires environment/Dockerfile. CasinoBench is
# manifest-backed: when run with --environment-manifest, the manifest's run
# `image` (ghcr.io/yiminnn/casinobench-base:2.0.6) is the sandbox target and THIS file is
# bypassed. It is a trivial passthrough so the per-game task carries no per-task
# build (the game is chosen at runtime by CASINOBENCH_GAME, not baked here), and
# so the non-manifest path still gets the working casino-service image.
FROM ghcr.io/yiminnn/casinobench-base:2.0.6
