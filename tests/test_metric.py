"""Tests for the cognitive load metric. They run offline on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability import (
    CognitiveLoad,
    Dataset,
    ExpectedPathLength,
    NumberOfFeaturesUsed,
    NumberOfSplits,
    NumberOfTrees,
    get_trees,
)


@pytest.fixture
def dataset() -> Dataset:
    X, y = make_classification(n_samples=300, n_features=8, n_informative=5, random_state=0)
    return Dataset(name="synthetic", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))


def test_stump_has_load_three(dataset):
    """One question: 1 split + 1 feature + 1 question per prediction + 0 trees to combine = 3."""
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(dataset.X, dataset.y)
    assert CognitiveLoad().score(stump, dataset) == 3


def test_load_is_sum_of_parts(dataset):
    forest = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0).fit(dataset.X, dataset.y)
    parts = [NumberOfSplits(), NumberOfFeaturesUsed(), ExpectedPathLength()]
    expected = sum(m.score(forest, dataset) for m in parts) + NumberOfTrees().score(forest, dataset) - 1
    assert CognitiveLoad().score(forest, dataset) == pytest.approx(expected)


def test_expected_path_length_matches_decision_paths(dataset):
    """The sample-weighted path length equals the average number of questions over the training data."""
    tree = DecisionTreeClassifier(max_depth=5, random_state=0).fit(dataset.X, dataset.y)
    nodes_per_sample = np.asarray(tree.decision_path(dataset.X).sum(axis=1)).ravel()
    assert ExpectedPathLength().score(tree, dataset) == pytest.approx((nodes_per_sample - 1).mean())


def test_deeper_trees_are_never_easier(dataset):
    loads = [
        CognitiveLoad().score(DecisionTreeClassifier(max_depth=d, random_state=0).fit(dataset.X, dataset.y), dataset)
        for d in range(1, 8)
    ]
    assert loads == sorted(loads)


def test_more_trees_means_more_load(dataset):
    small = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=0).fit(dataset.X, dataset.y)
    large = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0).fit(dataset.X, dataset.y)
    assert CognitiveLoad().score(small, dataset) < CognitiveLoad().score(large, dataset)


def test_boosting_with_three_classes_has_one_tree_per_class_and_round():
    X, y = make_classification(n_samples=200, n_features=6, n_informative=4, n_classes=3, random_state=0)
    model = GradientBoostingClassifier(n_estimators=4, max_depth=2, random_state=0).fit(X, y)
    assert len(get_trees(model)) == 4 * 3


def test_non_tree_models_are_rejected(dataset):
    model = LogisticRegression().fit(dataset.X, dataset.y)
    with pytest.raises(TypeError):
        CognitiveLoad().score(model, dataset)
