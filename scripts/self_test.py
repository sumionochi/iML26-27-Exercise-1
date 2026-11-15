"""Human self-test: can *you* still use a model by hand? This gives human evidence for the threshold.

Usage (takes about 15 minutes, run it in a terminal)::

    uv run python scripts/self_test.py

For each model, from small to large, the script prints the model, then shows a few Titanic passengers.
You predict by hand what the *model* says (not what really happened) and rate how well you understand
the model. Type ``q`` to give up on a model: that counts as "not interpretable" too.

Results go to ``results/self_test.csv`` and ``results/self_test.png``. ``run_experiment.py`` then reads
the CSV and draws your threshold into its plot.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

from iml_interpretability import CognitiveLoad, Dataset, get_models, get_trees, load_titanic

RESULTS = Path("results")
SEED = 0
N_PASSENGERS = 3
# Small to large. Gradient boosting is left out: predicting it by hand means adding up decimal numbers
# and applying a sigmoid, which is clearly beyond "interpretable" already at a few trees.
MODELS = ["tree_d1", "tree_d2", "tree_d3", "tree_d4", "tree_d5", "tree_d6", "forest_3_d2"]


def ask_prediction(prompt: str) -> str:
    """Ask until the answer is 0, 1 or q."""
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"0", "1", "q"}:
            return answer
        print("  Please type 0, 1 or q.")


def ask_rating() -> int:
    """Ask for a 1-5 understanding rating."""
    while True:
        answer = input("How well do you understand this model? 1 = not at all ... 5 = completely: ").strip()
        if answer in {"1", "2", "3", "4", "5"}:
            return int(answer)
        print("  Please type a number from 1 to 5.")


def show_model(name: str, model, feature_names: list[str]) -> list[str]:
    """Print every tree of the model and return the names of the features it uses."""
    trees = get_trees(model)
    used = sorted({feature_names[i] for tree in trees for i in tree.tree_.feature if i >= 0})
    print(f"\n{'=' * 80}\nMODEL: {name}  ({len(trees)} tree(s))")
    if len(trees) > 1:
        print("The forest predicts by MAJORITY VOTE of its trees.")
    for index, tree in enumerate(trees, start=1):
        if len(trees) > 1:
            print(f"\n--- tree {index} ---")
        print(export_text(tree, feature_names=feature_names, class_names=["died", "survived"]))
    return used


def run_model(name: str, dataset: Dataset, X_train, y_train, X_test, rng: np.random.Generator) -> dict:
    """Show one model, ask for predictions and a rating, and return one result row."""
    model = clone(get_models(seed=SEED)[name]).fit(X_train, y_train)
    used = show_model(name, model, dataset.feature_names)

    passengers = X_test.iloc[rng.choice(len(X_test), size=N_PASSENGERS, replace=False)]
    n_correct, seconds, gave_up = 0, [], False
    for number, (_, passenger) in enumerate(passengers.iterrows(), start=1):
        print(f"\nPassenger {number}: " + ", ".join(f"{feature}={passenger[feature]:g}" for feature in used))
        start = time.monotonic()
        answer = ask_prediction("What does the MODEL predict? 0 = died, 1 = survived, q = give up: ")
        seconds.append(time.monotonic() - start)
        if answer == "q":
            gave_up = True
            break
        truth = int(model.predict(passenger.to_frame().T)[0])
        n_correct += int(answer) == truth
        print("  correct" if int(answer) == truth else f"  wrong, the model says {truth}")

    rating = 1 if gave_up else ask_rating()
    return {
        "model": name,
        "cognitive_load": CognitiveLoad().score(model, dataset),
        "n_correct": n_correct,
        "n_asked": N_PASSENGERS,
        "gave_up": gave_up,
        "seconds_per_prediction": float(np.mean(seconds)),
        "rating": rating,
    }


def plot_self_test(results: pd.DataFrame, path: Path) -> None:
    """Rating and time per prediction against cognitive load."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, column, label in [(left, "rating", "understanding (1-5)"),
                              (right, "seconds_per_prediction", "seconds per prediction")]:
        trees = results[results["model"].str.startswith("tree")].sort_values("cognitive_load")
        forests = results[~results["model"].str.startswith("tree")]
        ax.plot(trees["cognitive_load"], trees[column], "o-", color="#2a7ab9", label="decision tree")
        ax.scatter(forests["cognitive_load"], forests[column], marker="s", color="#2f9e44", label="random forest",
                   zorder=3)
        for _, row in results.iterrows():
            ax.annotate(row["model"], (row["cognitive_load"], row[column]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
        ax.set_xlabel("cognitive load (mental steps)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    left.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    dataset = load_titanic()
    X_train, X_test, y_train, _ = train_test_split(
        dataset.X, dataset.y, test_size=0.25, random_state=SEED, stratify=dataset.y
    )
    rng = np.random.default_rng(SEED)
    print("Self-test: predict what each model says for a few passengers, then rate your understanding.")

    rows = [run_model(name, dataset, X_train, y_train, X_test, rng) for name in MODELS]
    results = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS / "self_test.csv", index=False)
    plot_self_test(results, RESULTS / "self_test.png")
    print("\n" + results.round(2).to_string(index=False))
    print(f"\nSaved {RESULTS / 'self_test.csv'} and {RESULTS / 'self_test.png'}")


if __name__ == "__main__":
    main()
