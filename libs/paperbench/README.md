# Preparedness Evals

This repository contains the code for multiple Preparedness evals that use nanoeval and alcatraz.

## System requirements

1. Python 3.11 (3.12 is untested; 3.13 will break [chz](https://github.com/openai/chz))

## Install pre-requisites

```bash
for proj in nanoeval alcatraz nanoeval_alcatraz; do
    pip install -e project/"$proj"
done
```

## Evals

- [PaperBench](./project/paperbench/README.md)
- SWELancer (Forthcoming)
- MLE-bench (Forthcoming)

## About This Repository
This repository is part of the official [OpenAI preparedness evaluations](https://github.com/openai/preparedness/tree/main).
It provides the implementation for several Preparedness evals that utilize nanoeval and alcatraz frameworks.

### ⚠️ Note
If you wish to run these evaluations as part of the OpenAI preparedness framework,
please replace the paperbench directory in the original preparedness repo with the project/paperbench from this repository.

This will allow you to seamlessly integrate the latest version of PaperBench into the OpenAI evaluation pipeline.
