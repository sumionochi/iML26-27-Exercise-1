"""Run the interpretability evaluation, store the results as CSV and draw the plots.

Usage::

    uv run python scripts/run_experiment.py

Outputs (in ``results/``):

- ``scores.csv``: one row per (dataset, model) with accuracy, cognitive load and its four parts.
- ``accuracy_vs_cognitive_load.png``: the main plot, one panel per dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from iml_interpretability import all_metrics, evaluate, get_models, load_all_datasets, model_family

RESULTS = Path("results")
SEED = 0

FAMILY_STYLE = {
    "tree": {"color": "#2a7ab9", "marker": "o", "label": "decision tree"},
    "forest": {"color": "#2f9e44", "marker": "s", "label": "random forest"},
    "boost": {"color": "#d9480f", "marker": "^", "label": "gradient boosting"},
}


def threshold_from_self_test(path: Path) -> float | None:
    """Read the human self-test and return the threshold of interpretability, or None if not run yet.

    A model *passes* if the person predicted every passenger correctly, did not give up and rated their
    understanding at least 3 of 5. Going from small to large models, the threshold is the cognitive load
    of the last model that passed before the first one that failed.
    """
    if not path.exists():
        return None
    test = pd.read_csv(path).sort_values("cognitive_load")
    passed = (test["n_correct"] == test["n_asked"]) & ~test["gave_up"] & (test["rating"] >= 3)
    threshold = None
    for load, ok in zip(test["cognitive_load"], passed, strict=True):
        if not ok:
            break
        threshold = float(load)
    return threshold


def summarize(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per dataset: best model within the threshold vs. best model overall (the price of interpretability)."""
    rows = []
    for name, subset in results.groupby("dataset", sort=False):
        best = subset.loc[subset["accuracy"].idxmax()]
        interpretable = subset[subset["cognitive_load"] <= threshold]
        best_ok = interpretable.loc[interpretable["accuracy"].idxmax()]
        rows.append({
            "dataset": name,
            "best_interpretable": best_ok["model"],
            "its_accuracy": best_ok["accuracy"],
            "its_load": best_ok["cognitive_load"],
            "best_overall": best["model"],
            "best_accuracy": best["accuracy"],
            "best_load": best["cognitive_load"],
            "accuracy_lost": best["accuracy"] - best_ok["accuracy"],
        })
    return pd.DataFrame(rows)


def plot_accuracy_vs_load(results: pd.DataFrame, path: Path, threshold: float | None = None) -> None:
    """One panel per dataset: test accuracy (y) against cognitive load (x, log scale)."""
    names = list(results["dataset"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4), sharey=False)
    for ax, name in zip(axes, names, strict=True):
        subset = results[results["dataset"] == name]
        for family, style in FAMILY_STYLE.items():
            rows = subset[subset["family"] == family]
            ax.scatter(rows["cognitive_load"], rows["accuracy"], color=style["color"], marker=style["marker"],
                       label=style["label"], s=40, zorder=3)
            if family == "tree":
                # Connect the trees in order of depth and label a few of them.
                ax.plot(rows["cognitive_load"], rows["accuracy"], color=style["color"], alpha=0.4, zorder=2)
                labelled: set[float] = set()  # deep trees can stop growing and land on the same point
                for _, row in rows.iterrows():
                    load = round(row["cognitive_load"], 3)
                    if row["model"] in {"tree_d1", "tree_d3", "tree_d5", "tree_full"} and load not in labelled:
                        labelled.add(load)
                        ax.annotate(row["model"].removeprefix("tree_"), (row["cognitive_load"], row["accuracy"]),
                                    textcoords="offset points", xytext=(4, -10), fontsize=8)
        if threshold is not None:
            # Shade the interpretable region (cognitive load 1 to threshold).
            ax.axvspan(1, threshold, color="#2a7ab9", alpha=0.08, zorder=1)
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1, zorder=1,
                       label=f"threshold ({threshold:.0f})")
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("cognitive load (mental steps, log scale)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    results = evaluate(
        datasets=load_all_datasets(),
        models=get_models(seed=SEED),
        metrics=all_metrics(),
        seed=SEED,
    )
    results.insert(2, "family", results["model"].map(model_family))

    threshold = threshold_from_self_test(RESULTS / "self_test.csv")

    RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS / "scores.csv", index=False)
    plot_accuracy_vs_load(results, RESULTS / "accuracy_vs_cognitive_load.png", threshold)

    pd.set_option("display.width", 200)
    print(results.round(3).to_string(index=False))

    if not (RESULTS / "self_test.csv").exists():
        print("\nNo self-test yet: run `uv run python scripts/self_test.py` to find your threshold.")
    elif threshold is None:
        print("\nIn the self-test not even the smallest model passed, so no threshold could be set.")
    else:
        summary = summarize(results, threshold)
        summary.to_csv(RESULTS / "summary.csv", index=False)
        print(f"\nThreshold of interpretability from the self-test: cognitive load <= {threshold:.1f}")
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
