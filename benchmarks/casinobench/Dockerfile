# Run image for the casinobench benchmark = the prebuilt base (agent backends +
# casino CLI + the in-sandbox casino-service). The manifest's `image` key
# (casinobench-base:latest) selects it as a prebuilt, so this Dockerfile is not
# rebuilt at run time — it exists so the sandbox's environment_dir validates and
# so `docker build -t casinobench-base:latest benchmarks/casinobench` works too.
# Build the base first: see docker/casinobench-base.Dockerfile + README.md.
FROM casinobench-base:latest
