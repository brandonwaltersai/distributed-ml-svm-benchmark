from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path: str, target: str = "IsBadBuy") -> pd.DataFrame:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")
    if df[target].nunique(dropna=True) != 2:
        raise ValueError(f"Target '{target}' must be binary.")
    return df.dropna(subset=[target]).copy()


def split_frame(
    df: pd.DataFrame,
    target: str = "IsBadBuy",
    test_size: float = 0.20,
    random_state: int = 42,
):
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical
