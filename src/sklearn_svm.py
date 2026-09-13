from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .data_prep import column_groups, load_csv, split_frame
from .metrics import evaluate_binary


def build_pipeline(X):
    numeric, categorical = column_groups(X)

    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))

    preprocess = ColumnTransformer(transformers=transformers)
    model = SVC(kernel="rbf", class_weight="balanced")
    return Pipeline([("preprocess", preprocess), ("model", model)])


def run(path: str, target: str = "IsBadBuy") -> dict[str, float]:
    df = load_csv(path, target)
    X_train, X_test, y_train, y_test = split_frame(df, target)
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    return evaluate_binary(y_test, pred).as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate an RBF SVM on a binary CSV dataset.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="IsBadBuy")
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.target), indent=2))


if __name__ == "__main__":
    main()
