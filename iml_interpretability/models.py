"""The tree models we compare.

To find the threshold of interpretability we need a *ladder* of models, from obviously interpretable
(a tree with one question) to obviously not interpretable (a forest of 100 deep trees). We use three
families from the tree model class:

- ``tree_d{k}``: one decision tree with at most ``k`` questions per path (k = 1..10), plus ``tree_full``
  without a depth limit.
- ``forest_{n}_d{k}``: a random forest of ``n`` trees, each with depth at most ``k`` (``full`` = no limit).
  Each tree is trained on a random sample of the data and the trees vote.
- ``boost_{n}_d{k}``: gradient boosting with ``n`` rounds of trees of depth ``k``. Each new tree corrects
  the mistakes of the previous ones and their outputs are added up.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

TREE_DEPTHS = range(1, 11)


def model_family(name: str) -> str:
    """Return the family of a model name from :func:`get_models`: ``tree``, ``forest`` or ``boost``."""
    return name.split("_")[0]


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return unfitted models, keyed by a unique name used in result tables.

    :param seed: Random seed passed to every model.
    :return: Mapping from model name to unfitted scikit-learn classifier.
    :rtype: dict[str, ClassifierMixin]
    """
    models: dict[str, ClassifierMixin] = {}

    for depth in TREE_DEPTHS:
        models[f"tree_d{depth}"] = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    models["tree_full"] = DecisionTreeClassifier(random_state=seed)

    for n_trees, depth in [(3, 2), (10, 3), (100, 3), (100, None)]:
        label = "full" if depth is None else depth
        models[f"forest_{n_trees}_d{label}"] = RandomForestClassifier(
            n_estimators=n_trees, max_depth=depth, random_state=seed
        )

    for n_rounds, depth in [(3, 1), (10, 2), (100, 3)]:
        models[f"boost_{n_rounds}_d{depth}"] = GradientBoostingClassifier(
            n_estimators=n_rounds, max_depth=depth, random_state=seed
        )

    return models
