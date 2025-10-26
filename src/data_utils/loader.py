"""
data_utils/loader.py

Utility functions to load pre-split Credit Card Fraud Detection datasets:
    - train.csv
    - val.csv
    - test.csv
"""

import pandas as pd
from typing import Tuple


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset (train/val/test) from a CSV file.

    Args:
        file_path (str): Full path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"[INFO] Successfully loaded: {file_path}")
        print(f"[INFO] Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        raise


def get_features_target(df: pd.DataFrame, target_col: str = "Class") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y).

    Args:
        df (pd.DataFrame): Loaded dataset.
        target_col (str): Name of the target column (default = 'Class').

    Returns:
        (X, y): Tuple of features and target.
    """
    if target_col not in df.columns:
        raise ValueError(f"[ERROR] Target column '{target_col}' not found in dataset!")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y
