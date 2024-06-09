from ucimlrepo import fetch_ucirepo
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder
import numpy as np


def encode_categorical(X: pd.DataFrame) -> pd.DataFrame:
    pass


def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], LabelEncoder]:
    if name == "wine":
        dataset = fetch_ucirepo(id=186)
        X = dataset.data.features
        y = dataset.data.targets
        y = y[np.logical_and(y!=3, y!=9)].dropna()
        X = X.loc[y.index]
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
    elif name == "adult":
        dataset = fetch_ucirepo(id=2)
        X = dataset.data.features
        y = dataset.data.targets
        y = y['income'].str.replace(".", "")
        X = X.dropna()
        y = y[X.index]
    else:
        raise ValueError("name must be one of {wine, adult}")
    categorical_columns: List[str] = []
    numerical_columns: List[str] = []
    for dtype, col_name in zip(dataset.variables.type, X.columns):
        if dtype in ["Continuous", 'Integer']:
            numerical_columns.append(col_name)
        elif dtype in ["Binary", "Categorical"]:
            categorical_columns.append(col_name)
        else:
            raise ValueError(f"Unknown col type {col_name}: {dtype}")
    le = LabelEncoder()
    y = pd.DataFrame(le.fit_transform(y))
    y.index = X.index
    return X, y, numerical_columns, categorical_columns, le
