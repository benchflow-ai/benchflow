import json

import pandas as pd


def read_model_results() -> dict:
    """
    Saves the aggregate metrics and token usage for each model from a set of paths
    """

    model_results = {}

    paths = [
        "experiments/judge_eval/judge_eval_results/simple_gpt-4o-2024-08-06_20250323_043512/results.json",
        "experiments/judge_eval/judge_eval_results/simple_gpt-4o-mini-2024-07-18_20250323_041827/results.json",
        "experiments/judge_eval/judge_eval_results/simple_o1-2024-12-17_20250323_003218/results.json",
        "experiments/judge_eval/judge_eval_results/simple_o1-mini-2024-09-12_20250323_014231/results.json",
        "experiments/judge_eval/judge_eval_results/simple_o3-mini-2025-01-31_20250322_232740/results.json",
    ]

    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)

        model_name = data["model_name"]
        reasoning_effort = (
            data.get("judge_kwargs", {}).get("completion_kwargs", {}).get("reasoning_effort", None)
        )
        key = f"{model_name}_{reasoning_effort}" if reasoning_effort else model_name
        model_results[key] = {
            "model_name": model_name,
            "reasoning_effort": reasoning_effort,
            "aggregate_metrics": data["aggregate_metrics"],
        }

    return model_results


def read_random_baseline_results():
    with open(
        "experiments/judge_eval/judge_eval_results/random_20250323_044950/results.json", "r"
    ) as f:
        data = json.load(f)

        random_baseline_results = data["aggregate_metrics"]

    return random_baseline_results


def make_performance_table(model_results, random_baseline_results):
    model_to_overall = {
        model: stats["aggregate_metrics"]["f1"] for model, stats in model_results.items()
    }
    model_to_code_dev = {
        model: stats["aggregate_metrics"]["stratified"]["Code Development"]["f1"]
        for model, stats in model_results.items()
    }
    model_to_code_exec = {
        model: stats["aggregate_metrics"]["stratified"]["Code Execution"]["f1"]
        for model, stats in model_results.items()
    }
    model_to_res_analysis = {
        model: stats["aggregate_metrics"]["stratified"]["Result Analysis"]["f1"]
        for model, stats in model_results.items()
    }

    models_sorted = [
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "o1-mini-2024-09-12_high",
        "o1-2024-12-17_high",
        "o3-mini-2025-01-31_high",
    ]

    # make table, with rows as models and columns being overall, code dev, code exec, and res analysis
    data = []
    for model in models_sorted:
        data.append(
            {
                "Model": model,
                "Overall": model_to_overall.get(model),
                "Code Development": model_to_code_dev.get(model),
                "Code Execution": model_to_code_exec.get(model),
                "Result Analysis": model_to_res_analysis.get(model),
            }
        )

    # Append random baseline row
    baseline = {
        "Model": "Random Baseline",
        "Overall": random_baseline_results.get("f1"),
        "Code Development": random_baseline_results.get("stratified", {})
        .get("Code Development", {})
        .get("f1"),
        "Code Execution": random_baseline_results.get("stratified", {})
        .get("Code Execution", {})
        .get("f1"),
        "Result Analysis": random_baseline_results.get("stratified", {})
        .get("Result Analysis", {})
        .get("f1"),
    }
    data.append(baseline)

    df = pd.DataFrame(data)
    df.to_csv("experiments/judge_eval/tables/perf_table.csv", index=False)
    print("Saved performance table to 'experiments/judge_eval/tables/perf_table.csv'")


# New function to create separate metrics tables for each model


def make_metrics_tables(model_results):
    for model, stats in model_results.items():
        overall_metrics = stats["aggregate_metrics"]
        metrics = {}
        metrics["Overall"] = {
            "accuracy": overall_metrics.get("accuracy"),
            "precision": overall_metrics.get("precision"),
            "recall": overall_metrics.get("recall"),
            "f1": overall_metrics.get("f1"),
        }
        stratified = overall_metrics.get("stratified", {})
        for category in ["Code Development", "Code Execution", "Result Analysis"]:
            category_metrics = stratified.get(category, {})
            metrics[category] = {
                "accuracy": category_metrics.get("accuracy"),
                "precision": category_metrics.get("precision"),
                "recall": category_metrics.get("recall"),
                "f1": category_metrics.get("f1"),
            }
        df = pd.DataFrame.from_dict(metrics, orient="index")
        output_path = f"experiments/judge_eval/tables/metrics_table_{model}.csv"
        df.to_csv(output_path, index=True)
        print(f"Saved metrics table for model {model} to '{output_path}'")


def main():
    model_results = read_model_results()
    random_baseline_results = read_random_baseline_results()

    make_performance_table(model_results, random_baseline_results)
    make_metrics_tables(model_results)


if __name__ == "__main__":
    main()
