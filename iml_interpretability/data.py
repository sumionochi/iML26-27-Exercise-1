"""Dataset loading.

Every dataset is returned as a :class:`Dataset` with numeric, imputed features so that all
models in :mod:`iml_interpretability.models` can be fit on it directly. We use four classification
datasets that differ in size, number of features and number of classes:

- titanic: did a passenger survive? (1309 rows, OpenML)
- credit: is a loan applicant a good or bad risk? (1000 rows, OpenML "credit-g")
- breast_cancer: is a tumour benign or malignant? (569 rows, built into scikit-learn)
- wine: which of 3 cultivars is a wine from? (178 rows, 3 classes, built into scikit-learn)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_openml

# OpenML downloads are cached here, relative to the current working directory.
DATA_HOME = Path("data")


@dataclass(frozen=True)
class Dataset:
    """A preprocessed tabular classification dataset.

    :param name: Human-readable dataset name, used in result tables.
    :param X: Numeric feature matrix without missing values.
    :param y: Target vector.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


# Columns that leak the target (boat = lifeboat, body = body identification number) or are
# free-text identifiers without a meaningful encoding.
_TITANIC_DROP_COLUMNS = ["boat", "body", "home.dest", "name", "ticket", "cabin"]


def load_titanic(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the Titanic dataset from OpenML (data id 40945).

    The download is cached in ``data_home``, so network access is only needed on the first call.

    :param data_home: Cache directory for the download, relative paths are resolved against the
        current working directory.
    :return: The preprocessed Titanic dataset.
    :rtype: Dataset
    """
    # The default pandas ARFF parser crashes on Titanic's integer-valued categorical columns.
    bunch = fetch_openml("titanic", version=1, as_frame=True, parser="liac-arff", data_home=data_home)
    X = bunch.data.drop(columns=_TITANIC_DROP_COLUMNS)
    y = bunch.target.astype(int).rename("survived")

    numeric_columns = X.select_dtypes(include="number").columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    return Dataset(name="titanic", X=X, y=y)


def load_credit(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the German credit dataset from OpenML ("credit-g"): is a loan applicant a good or bad risk?

    1000 applicants, 20 features (7 numeric, 13 categorical). Categorical features are one-hot encoded.

    :param data_home: Cache directory for the download.
    :return: The preprocessed credit dataset, target 1 = good risk.
    :rtype: Dataset
    """
    bunch = fetch_openml("credit-g", version=1, as_frame=True, data_home=data_home)
    X = pd.get_dummies(bunch.data, drop_first=True, dtype=float)
    y = (bunch.target == "good").astype(int).rename("good_credit")
    return Dataset(name="credit", X=X, y=y)


def load_breast_cancer() -> Dataset:
    """Load scikit-learn's built-in breast cancer dataset (569 tumours, 30 numeric features, no download).

    :return: The breast cancer dataset, target 1 = benign.
    :rtype: Dataset
    """
    bunch = datasets.load_breast_cancer(as_frame=True)
    return Dataset(name="breast_cancer", X=bunch.data, y=bunch.target.rename("benign"))


def load_wine() -> Dataset:
    """Load scikit-learn's built-in wine dataset (178 wines, 13 numeric features, 3 classes, no download).

    :return: The wine dataset, target = cultivar (0, 1 or 2).
    :rtype: Dataset
    """
    bunch = datasets.load_wine(as_frame=True)
    return Dataset(name="wine", X=bunch.data, y=bunch.target.rename("cultivar"))


def load_all_datasets(data_home: str | Path = DATA_HOME) -> list[Dataset]:
    """Load every dataset used in the experiments.

    :param data_home: Cache directory for OpenML downloads.
    :return: Titanic, credit, breast cancer and wine.
    :rtype: list[Dataset]
    """
    return [load_titanic(data_home), load_credit(data_home), load_breast_cancer(), load_wine()]
