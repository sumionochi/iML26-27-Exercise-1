# iML 26/27 – Exercise 1: Cognitive Load of Tree Models

We measure how interpretable a tree model is by counting the **mental steps** a person needs to understand
it and to make a prediction with it by hand. The metric works for the whole tree model class: decision trees,
random forests and gradient-boosted trees. We apply it to 18 models on 4 datasets and use a human self-test
to find the **threshold of interpretability**. This submission goes for **Task 1 (1 point)**.

## The metric: cognitive load

A tree model consists of one or more binary decision trees. To work with it, a person has to:

| Part | What the person does | How we count it |
|---|---|---|
| `S` | read every rule (global understanding) | number of yes/no questions (split nodes) in all trees |
| `F` | know what every feature means (vocabulary) | number of distinct features used anywhere |
| `Q` | follow one prediction (local simulation) | questions answered from root to leaf, summed over all trees, weighted by the share of training samples on each path |
| `T - 1` | combine the trees' answers (ensembles only) | number of trees minus one |

**Cognitive load = S + F + Q + (T − 1)**, in mental steps. **Lower means more interpretable.**

Why this goes beyond a simple size measure:

- It combines **global** understanding (`S`, `F`) with **local** simulation (`Q`), the two main ways people
  use an interpretable model.
- `Q` depends on the data: a tree whose common cases are answered after one question is easier to use than
  an equally large tree where every case needs many questions.
- It works across model families: a forest pays for every tree it adds (`S`, `Q`) and for voting (`T − 1`).
- Depth alone is misleading: on breast cancer and wine, trees stop growing at depth 4–5, so "depth 10" there
  is as easy as "depth 5". The cognitive load notices this, while a depth limit does not.

All parts are counts of things a person processes, so they share one unit and are weighted equally.
A depth-1 tree (one question) has a load of exactly 3: 1 split + 1 feature + 1 question to answer.

## The threshold of interpretability

`scripts/self_test.py` shows models of growing size (decision trees of depth 1–6 and a 3-tree forest) on
Titanic. For each one, the person predicts by hand what the model says for 3 passengers and rates their
understanding from 1 to 5. A model **passes** if all 3 predictions are right, the person didn't give up, and
the rating is at least 3. The threshold is the cognitive load of the last model that passed before the first
one failed.

<!-- TODO: fill in after running the self-test -->
**Our result:** threshold = _TODO_ mental steps.

## Experiments

- **Datasets** (`iml_interpretability/data.py`): Titanic (survival), German credit (credit risk), breast
  cancer (benign or malignant), wine (3 cultivars). They differ in size (178–1309 rows), number of
  features (8–48 after encoding) and number of classes.
- **Models** (`iml_interpretability/models.py`): decision trees of depth 1–10 and unlimited, random forests
  (3×depth 2, 10×depth 3, 100×depth 3, 100×unlimited) and gradient boosting (3×depth 1, 10×depth 2,
  100×depth 3).
- **Evaluation** (`iml_interpretability/evaluate.py`): 75/25 train/test split (seed 0), test accuracy and
  the metric with all its parts for every (dataset, model) pair.

## Reproduce

```bash
uv sync --extra dev
uv run pytest                               # 10 tests
uv run python scripts/self_test.py          # interactive, about 15 min, writes results/self_test.csv
uv run python scripts/run_experiment.py     # writes results/scores.csv, summary.csv and the plot
```

`run_experiment.py` reads `results/self_test.csv` if it exists and draws the threshold into
`results/accuracy_vs_cognitive_load.png`. Titanic and credit are downloaded from OpenML on the first run and
cached in `data/`.

## Repository structure

```
iml_interpretability/
├── data.py        # the 4 datasets
├── models.py      # the 18 models (trees, forests, boosting)
├── metric.py      # CognitiveLoad and its parts S, F, Q, T
└── evaluate.py    # fits all models on all datasets and computes accuracy + metrics (from the template)
scripts/
├── run_experiment.py   # main experiment + plot
└── self_test.py        # human self-test for the threshold
tests/                  # pytest tests for the metric and the pipeline
presentation/           # presentation.pdf
```

## Limitations

- Equal weights for the four parts are a simplifying assumption; a reading step and a lookup step may not
  be equally hard.
- The self-test has one participant, few passengers per model, and shows models from small to large, so
  practice effects are possible.
- Accuracy comes from a single train/test split; small datasets (wine has 45 test rows) are noisy.

## AI statement

<!-- TODO: write this yourself. Say which AI tool you used (e.g. Claude Code), for what (which files or
steps), how you checked the output, and what you did yourself. -->
_TODO_
