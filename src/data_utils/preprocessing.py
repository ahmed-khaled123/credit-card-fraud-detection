"""
src/data_utils/preprocessing.py

Utilities for preprocessing credit-card fraud datasets:
- scaling (StandardScaler or RobustScaler)
- balancing training set with SMOTE
- save/load scaler artifact with joblib
- optional: save processed datasets to CSV

Usage pattern:
  1. load train/val/test via loader.load_dataset()
  2. call fit_scaler_and_transform_train() to get X_train_res, y_train_res and save scaler
  3. call transform_with_saved_scaler() on validation/test to get scaled X_val, X_test
"""

from typing import Tuple, Optional
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
import joblib
scaler_artifact_path = r"G:\ML mostafa saad\slides\my work\2 Credit Card Fraud Detection\models/scaler.joblib"

# -----------------------------
# Helper: choose scaler class
# -----------------------------
def _get_scaler(kind: str = "standard"):
    """
    Return an instance of scaler based on 'kind'.
    kind: "standard" or "robust"
    """
    kind = kind.lower()
    if kind == "standard":
        return StandardScaler()
    elif kind == "robust":
        return RobustScaler()
    else:
        raise ValueError("Unsupported scaler kind. Use 'standard' or 'robust'.")


# -----------------------------
# Fit scaler on X_train and transform train/val/test
# -----------------------------
def fit_scaler_and_transform_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scaler_kind: str = "standard",
    apply_smote: bool = True,
    smote_random_state: int = 42,
    save_scaler_path: Optional[str] = r"G:\ML mostafa saad\slides\my work\2 Credit Card Fraud Detection\models/scaler.joblib",
    columns_to_scale: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Fit a scaler on X_train, transform X_train, optionally apply SMOTE on the scaled train,
    and save the scaler artifact.

    Args:
        X_train: DataFrame of training features (raw).
        y_train: Series training labels.
        scaler_kind: "standard" or "robust".
        apply_smote: whether to apply SMOTE to the scaled X_train.
        smote_random_state: random_state for SMOTE.
        save_scaler_path: path to save the fitted scaler (joblib). If None, do not save.
        columns_to_scale: list of columns to scale. If None, scale all columns.

    Returns:
        X_train_res (np.ndarray): scaled (and possibly SMOTE-resampled) training features
        y_train_res (np.ndarray): corresponding labels
        scaler (object): fitted scaler instance
    """
    # choose columns
    cols = columns_to_scale if columns_to_scale is not None else list(X_train.columns)

    scaler = _get_scaler(scaler_kind)
    X_train_sub = X_train[cols].values  # keep order consistent

    # fit scaler on train only
    X_train_scaled = scaler.fit_transform(X_train_sub)

    # apply SMOTE only on training set (after scaling)
    if apply_smote:
        sm = SMOTE(random_state=smote_random_state)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train.values)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train.values

    # save scaler
    if save_scaler_path:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        joblib.dump({"scaler": scaler, "columns": cols}, save_scaler_path)
        print(f"[INFO] Scaler saved to: {save_scaler_path}")

    return X_train_res, y_train_res, scaler


# -----------------------------
# Transform validation/test with saved scaler
# -----------------------------
def transform_with_saved_scaler(
    X: pd.DataFrame, scaler_path: str = "models/scaler.joblib"
) -> np.ndarray:
    """
    Load scaler from joblib and transform provided DataFrame.

    Args:
        X: DataFrame to transform.
        scaler_path: path to the saved scaler artifact.

    Returns:
        X_scaled: np.ndarray scaled in the same way as training.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler artifact not found at: {scaler_path}")

    artifact = joblib.load(scaler_path)
    scaler = artifact["scaler"]
    cols = artifact["columns"]

    # ensure columns exist and in same order
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"The following expected columns are missing: {missing}")

    X_sub = X[cols].values
    X_scaled = scaler.transform(X_sub)
    return X_scaled


# -----------------------------
# Optional: save processed arrays to CSV
# -----------------------------
def save_processed_dataset(
    X: np.ndarray,
    y: np.ndarray,
    cols: Optional[list] = None,
    out_csv_path: Optional[str] = None,
):
    """
    Save processed dataset (numpy array) to CSV. If out_csv_path is None, do nothing.

    Args:
        X: numpy array features
        y: numpy array labels
        cols: column names for features (optional). If None, will use generic names.
        out_csv_path: path to save CSV
    """
    if out_csv_path is None:
        return

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    if cols is None:
        cols = [f"f{i}" for i in range(X.shape[1])]
    df_out = pd.DataFrame(X, columns=cols)
    df_out["Class"] = y
    df_out.to_csv(out_csv_path, index=False)
    print(f"[INFO] Processed dataset saved to: {out_csv_path}")
