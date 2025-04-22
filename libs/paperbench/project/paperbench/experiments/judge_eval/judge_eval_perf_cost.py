import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import container
from matplotlib.ticker import MultipleLocator

MODEL_COST_PER_TOKEN = {
    "gpt-4o-2024-08-06": {"in": 0.0000025, "out": 0.00001},
    "gpt-4o": {"in": 0.0000025, "out": 0.00001},
    "gpt-4o-mini": {"in": 0.00000015, "out": 0.0000006},
    "gpt-4o-mini-2024-07-18": {"in": 0.00000015, "out": 0.0000006},
    "o1": {"in": 0.000015, "out": 0.00006},
    "o1-2024-12-17": {"in": 0.000015, "out": 0.00006},
    "o1-mini": {"in": 0.0000011, "out": 0.0000044},
    "o1-mini-2024-09-12": {"in": 0.0000011, "out": 0.0000044},
    "o3-mini": {"in": 0.0000011, "out": 0.0000044},
    "o3-mini-2025-01-31": {"in": 0.0000011, "out": 0.0000044},
}

MODEL_NAME_TO_LABEL = {
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4o-2024-08-06": "gpt-4o",
    "o1-mini-2024-09-12": "o1-mini",
    "o1-2024-12-17": "o1",
    "o3-mini-2025-01-31": "o3-mini",
}

MODELS_SORTED = [
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    # "o1-mini-2024-09-12_medium",
    # "o1-mini-2024-09-12_low",
    "o1-mini-2024-09-12_high",
    # "o1-2024-12-17_medium",
    # "o1-2024-12-17_low",
    "o1-2024-12-17_high",
    # "o3-mini-2025-01-31_medium",
    # "o3-mini-2025-01-31_low",
    "o3-mini-2025-01-31_high",
]


def compute_cost(token_usage, model_cost_per_token) -> float:
    """
    Computes the total cost of using a model given token usage and cost per token
    """
    total_cost = 0
    for model in token_usage.keys():
        in_toks_cost = token_usage[model]["in"] * model_cost_per_token[model]["in"]
        out_toks_cost = token_usage[model]["out"] * model_cost_per_token[model]["out"]
        total_cost += in_toks_cost + out_toks_cost
    return total_cost


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
            "aggregate_token_usage": data["aggregate_token_usage"],
        }

    return model_results


def read_random_baseline_results():
    with open(
        "experiments/judge_eval/judge_eval_results/random_20250323_044950/results.json", "r"
    ) as f:
        data = json.load(f)

        random_baseline_results = data["aggregate_metrics"]

    return random_baseline_results


def plot_perf_cost(model_results, random_baseline_results):
    model_to_f1 = {
        model: stats["aggregate_metrics"]["f1"] for model, stats in model_results.items()
    }

    model_to_cost_per_rubric = {
        model: compute_cost(stats["aggregate_token_usage"]["mean"], MODEL_COST_PER_TOKEN)
        for model, stats in model_results.items()
    }

    model_name_to_marker = {
        "gpt-4o-mini-2024-07-18": "o",  # Circle
        "gpt-4o-2024-08-06": "s",  # Square
        "o1-mini-2024-09-12": "^",  # Triangle
        "o1-2024-12-17": "D",  # Diamond
        "o3-mini-2025-01-31": "v",  # triangle down
    }

    model_name_to_color = {
        "gpt-4o-mini-2024-07-18": "blue",
        "gpt-4o-2024-08-06": "green",
        "o1-mini-2024-09-12": "red",
        "o1-2024-12-17": "purple",
        "o3-mini-2025-01-31": "orange",
    }

    reasoning_effort_to_marker_size = {
        None: 4,
        "low": 1,
        "medium": 2,
        "high": 4,
    }

    plt.rcParams.update({"font.size": 7})

    f, ax = plt.subplots(1, 1, figsize=(6.75133 / 1.5, 2.75), dpi=300)

    # Track which models we've already added to legend
    legend_models = set()

    for model in MODELS_SORTED:
        model_name = model_results[model]["model_name"]
        reasoning_effort = model_results[model]["reasoning_effort"]

        # to avoid duplicated legend entries
        label = MODEL_NAME_TO_LABEL[model_name] if model_name not in legend_models else None

        ax.scatter(
            model_to_cost_per_rubric[model],
            model_to_f1[model],
            marker=model_name_to_marker[model_name],
            color=model_name_to_color[model_name],
            label=label,
            s=reasoning_effort_to_marker_size[reasoning_effort] ** 2 * 2,
            linewidth=0.2,
            edgecolor="black",
        )

        legend_models.add(model_name)
    ax.axhline(
        random_baseline_results["f1"],
        color="red",
        label="random baseline",
        linewidth=0.5,
        linestyle="--",
    )
    ax.scatter(12 * 100, 1, marker="*", color="black", label="expert human", s=4**2 * 2)

    ax.set_xlabel("Average SimpleJudge Cost Per Paper [USD]")
    ax.set_ylabel("Performance on JudgeEval [F1]")
    ax.set_xscale("log")

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, loc="upper left")

    ax.grid(axis="y", which="major", linewidth=0.1, alpha=0.5)
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(axis="y", which="minor", linewidth=0.05, alpha=0.5)

    f.tight_layout()

    plt.savefig(
        "experiments/judge_eval/perf_cost.pdf", bbox_inches="tight", dpi=400, pad_inches=0.01
    )
    print("Saved plot to 'experiments/judge_eval/perf_cost.pdf'")


def create_metrics_table(model_results):
    """
    Creates a DataFrame with performance and cost metrics for each model
    """
    data = []
    # Use the same model order as in the plot
    for model in MODELS_SORTED:
        stats = model_results[model]
        model_name = stats["model_name"]
        reasoning_effort = stats["reasoning_effort"]
        f1_score = stats["aggregate_metrics"]["f1"]
        cost = compute_cost(stats["aggregate_token_usage"]["mean"], MODEL_COST_PER_TOKEN)

        # Use the same model name mapping as in the plot
        display_name = MODEL_NAME_TO_LABEL[model_name]

        data.append(
            {
                "Model": display_name,
                "Reasoning Effort": reasoning_effort if reasoning_effort else "default",
                "F1 Score": f1_score,
                "Avg Cost Per Paper (USD)": cost,
            }
        )

    df = pd.DataFrame(data)

    # No need to sort since we're using MODELS_SORTED order

    # Save to CSV
    output_path = "experiments/judge_eval/tables/perf_cost_table.csv"
    df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved metrics table to '{output_path}'")

    return df


def main():
    model_results = read_model_results()
    random_baseline_results = read_random_baseline_results()

    # Create and save metrics table
    create_metrics_table(model_results)

    # Create performance vs cost plot
    plot_perf_cost(model_results, random_baseline_results)


if __name__ == "__main__":
    main()
