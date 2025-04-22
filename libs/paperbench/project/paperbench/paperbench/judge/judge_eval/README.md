# Judge Evaluation (JudgeEval)

We have constructed a mini-eval to evaluate the performance of judges,
JudgeEval.

We manually grade a set of example submissions and treat these manual grades as
ground truth. We then run the evaluated judge on these same examples and compare
the judge's grading to the ground truth.

Judges are evaluated on accuracy, precision, recall, and F1 score,
macro-averaged across all the binary nodes across all the examples. We pay
particular attention to the F1 score given that the classes are imbalanced.

We note that JudgeEval functions independently of the reproduction step in
PaperBench: the reproduce.sh is not run as part of JudgeEval. The examples in
JudgeEval are assumed to be in the state they would be after running their
respective reproduce.sh scripts, if their reproduce.sh scripts exist.

## Dataset

JudgeEval consists of 5 submissions accompanied with their manually graded
'expected results'.

More specifically, each example consists of:

- a `submission.tar` file, which when untarred corresponds to the example
  submission folder to be graded by the judge.
- a `grading/expected_result.json` file, which corresponds to the manually
  graded rubric for that submission, with binary scores for each leaf node
  accompanied by an explanation.

We curate the following 5 examples, available under `data/judge_eval/`:

1. `stay-on-topic-with-classifier-free-guidance/0`: An incomplete submission for
   the reproduction of `stay-on-topic-with-classifier-free-guidance` with only
   some parts of the paper implemented and reproduced. The submission includes a
   reproduce.sh and is in the state it would be after running it.
2. `rice/0`: The authors' repository for `rice`. This submission does not come
   with a reproduce.sh script.
3. `all-in-one/0`: The authors' repository for `all-in-one`. This submission does
   not come with a reproduce.sh script.
4. `semantic-self-consistency/0`: An incomplete submission for the reproduction
   of `semantic-self-consistency` with only some parts of the paper implemented
   and reproduced. The submission includes a reproduce.sh and is in the state it
   would be after running it.
5. `pinn/0`: The authors' repository for `pinn`. This submission has been
   modified such that it comes with a `reproduce.sh` and is in the state it
   would be after running it. This is a close to complete reproduction of the
   paper.

We note that the graded rubrics in JudgeEval may differ from their official
counterparts in PaperBench. This is because we further polished the rubrics in
PaperBench separately to our JudgeEval experiments. Nevertheless, JudgeEval
functions independently of the rubrics in PaperBench, so the differences are not
consequential.

### Downloading Benchmark Data

We include most of the JudgeEval examples in the repository.

We however cannot redistribute some examples, which are therefore missing the
`submission.tar` file.

For these examples, we have the users obtain the missing submission tars
themselves.

To download and create the missing submission tars:

```bash
python download_data.py
```

### Tar files

To run JudgeEval, the submission files must be untarred.

To untar the submission files, run the following command from the
`data/judge_eval/` directory:

```bash
find . -name "submission.tar" -execdir tar -xvf {} \;
```

The .tar files were created by running the following command from the
appropriate example directory (e.g.
`stay-on-topic-with-classifier-free-guidance/0`):

```bash
tar -cvf submission.tar submission
```

## Running the evaluation

```bash
python paperbench/scripts/run_judge_eval.py -j <JUDGE_TYPE> -m <MODEL_NAME> -o <OUTPUT_DIR>
```

where `<OUTPUT_DIR>` is the directory to write the evaluation results to.

## JudgeEval from the PaperBench paper

The code and results for the JudgeEval portion in the PaperBench paper can be
found in `experiments/judge_eval/`.

Simply run (note this takes a while and can be expensive!):

```bash
bash experiments/judge_eval/run_judge_eval_sweep.sh
```

Then update the paths in `judge_eval_perf_cost.py` and
`judge_eval_per_tables.py` to regenerate the plots and tables from the paper.

## Attributions

We re-distribute (and modify) the following repositories for the judge
evaluation:

- [pratikrathore8/opt_for_pinns](https://github.com/pratikrathore8/opt_for_pinns):
  the author-provided repository for the paper "Challenges in Training PINNs: A
  Loss Landscape Perspective" by Rathore et al. (2024). Attribution is given to
  Pratik Rathore (@pratikrathore8) and Weimu Lei (@leiweimu). We modify this to
  include a functioning reproduce.sh. We obtain author permission to modify and
  redistribute the repository for JudgeEval.
- [mackelab/simformer](https://github.com/mackelab/simformer). Licensed under
  MIT. No modifications were made to this repository.
