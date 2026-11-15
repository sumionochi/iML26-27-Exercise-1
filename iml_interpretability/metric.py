"""Interpretability metrics for tree-based models.

Our metric, :class:`CognitiveLoad`, counts the *mental steps* a person needs to understand a tree model
and use it by hand. It works for every model in the tree model class: a single decision tree, a random
forest and a gradient-boosted ensemble, because all three are made of the same building block, a
binary decision tree, and the metric is computed from those trees.

The idea: a human who wants to work with a tree model has to do four kinds of work.

1. **Read the rules** (global understanding): look at every yes/no question in every tree once.
   Counted by :class:`NumberOfSplits` (``S``).
2. **Know the vocabulary**: understand what every feature the model talks about means.
   Counted by :class:`NumberOfFeaturesUsed` (``F``).
3. **Follow one prediction** (local simulation): answer the questions on the path from the root to a
   leaf, in every tree. Measured by :class:`ExpectedPathLength` (``Q``), weighted by how many training
   samples take each path, so it describes how the model is actually used, not just its shape.
4. **Combine the trees**: an ensemble of ``T`` trees gives ``T`` answers that still have to be added up
   or voted on, which costs ``T - 1`` extra steps (0 for a single tree). See :class:`NumberOfTrees`.

``CognitiveLoad = S + F + Q + (T - 1)``. **Lower means more interpretable.** Each part is a count of
things a person has to process, so they share one unit (mental steps) and can be added. We weight all
parts equally because we have no evidence that one kind of step is harder than another.

:class:`RelativeNumberOfLeaves` is the course template's example and is kept for reference only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier

from iml_interpretability.data import Dataset


class InterpretabilityMetric(ABC):
    """Base class for a metric that quantifies how interpretable a fitted model is."""

    #: Column name under which the score is reported.
    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score of a fitted model.

        :param model: A model fitted on ``dataset``.
        :param dataset: The dataset the model was fitted on.
        :return: The interpretability score. Document whether higher means more or less interpretable.
        :rtype: float
        """


# ----------------------------------------------------------------------------------------------------
# Helpers: look inside the trees of a fitted model
# ----------------------------------------------------------------------------------------------------


def get_trees(model: ClassifierMixin) -> list[BaseDecisionTree]:
    """Return the individual decision trees a fitted tree model consists of.

    - A decision tree is one tree.
    - A random forest stores its trees in ``estimators_`` (a list).
    - Gradient boosting stores them in ``estimators_`` as a 2-D array of shape
      ``(n_rounds, n_classes_or_1)``: with 3 classes it builds 3 trees per round.

    :param model: A fitted decision tree, random forest or gradient-boosting model.
    :return: All trees of the model.
    :rtype: list[BaseDecisionTree]
    :raises TypeError: If the model is not a tree model.
    """
    if isinstance(model, BaseDecisionTree):
        return [model]
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        raise TypeError(f"Expected a fitted tree model, got {type(model).__name__}")
    trees = list(np.ravel(estimators))
    if not all(isinstance(tree, BaseDecisionTree) for tree in trees):
        raise TypeError(f"{type(model).__name__} does not consist of decision trees")
    return trees


def tree_n_splits(tree: BaseDecisionTree) -> int:
    """Number of yes/no questions (internal nodes) in one tree."""
    return int(tree.tree_.node_count - tree.tree_.n_leaves)


def tree_features(tree: BaseDecisionTree) -> set[int]:
    """Indices of the features one tree asks about. Leaves are marked with a negative feature index."""
    features = tree.tree_.feature
    return set(features[features >= 0].tolist())


def tree_expected_path_length(tree: BaseDecisionTree) -> float:
    """Average number of questions answered for one prediction, weighted by training samples per leaf.

    We walk through the tree from the root and note the depth of every leaf. Each leaf then counts as
    much as the share of training samples that ended up in it. Example: if 90% of the samples reach a
    leaf at depth 1 and 10% a leaf at depth 5, the expected path length is 0.9 * 1 + 0.1 * 5 = 1.4.
    """
    structure = tree.tree_
    left, right = structure.children_left, structure.children_right
    weights = structure.weighted_n_node_samples

    total = 0.0
    stack = [(0, 0)]  # (node id, depth); node 0 is the root
    while stack:
        node, depth = stack.pop()
        if left[node] == right[node]:  # both -1: this node is a leaf
            total += depth * weights[node]
        else:
            stack.append((left[node], depth + 1))
            stack.append((right[node], depth + 1))
    return float(total / weights[0])


# ----------------------------------------------------------------------------------------------------
# The parts of our metric. Each one is also reported as its own column, which helps explain results.
# ----------------------------------------------------------------------------------------------------


class NumberOfTrees(InterpretabilityMetric):
    """``T``: how many trees a person has to look at (1 for a decision tree)."""

    name = "n_trees"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        return float(len(get_trees(model)))


class NumberOfSplits(InterpretabilityMetric):
    """``S``: total number of yes/no questions in all trees, i.e. the rules a person has to read."""

    name = "n_splits"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        return float(sum(tree_n_splits(tree) for tree in get_trees(model)))


class NumberOfFeaturesUsed(InterpretabilityMetric):
    """``F``: number of distinct features asked about anywhere in the model (the vocabulary)."""

    name = "n_features_used"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        used: set[int] = set()
        for tree in get_trees(model):
            used |= tree_features(tree)
        return float(len(used))


class ExpectedPathLength(InterpretabilityMetric):
    """``Q``: questions answered to make one prediction by hand, summed over all trees."""

    name = "path_length"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        return float(sum(tree_expected_path_length(tree) for tree in get_trees(model)))


class CognitiveLoad(InterpretabilityMetric):
    """Our interpretability metric: ``S + F + Q + (T - 1)`` mental steps. Lower = more interpretable.

    See the module docstring for the reasoning behind each part.
    """

    name = "cognitive_load"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        n_trees = NumberOfTrees().score(model, dataset)
        return (
            NumberOfSplits().score(model, dataset)
            + NumberOfFeaturesUsed().score(model, dataset)
            + ExpectedPathLength().score(model, dataset)
            + (n_trees - 1)
        )


def all_metrics() -> list[InterpretabilityMetric]:
    """The metrics reported in our experiments: the four parts and the combined cognitive load."""
    return [NumberOfTrees(), NumberOfSplits(), NumberOfFeaturesUsed(), ExpectedPathLength(), CognitiveLoad()]


# ----------------------------------------------------------------------------------------------------
# Template example (kept for reference, not used in our experiments)
# ----------------------------------------------------------------------------------------------------


class RelativeNumberOfLeaves(InterpretabilityMetric):
    """Number of leaves of a decision tree relative to the maximum possible for its depth.

    This is the course template's example. It only works for single decision trees.
    """

    name = "relative_n_leaves"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(f"{type(self).__name__} only supports DecisionTreeClassifier, got {type(model).__name__}")
        return model.get_n_leaves() / 2 ** model.get_depth()
