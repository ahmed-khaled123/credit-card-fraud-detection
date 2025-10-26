# example usage
import sys
import os

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_utils.loader import load_dataset, get_features_target
from src.data_utils.preprocessing import (
    fit_scaler_and_transform_train,
    transform_with_saved_scaler,
    save_processed_dataset,
)

train_path = r"G:\ML mostafa saad\slides\my work\2 Credit Card Fraud Detection\data\split\train.csv"
val_path = r"G:\ML mostafa saad\slides\my work\2 Credit Card Fraud Detection\data\split\val.csv"
test_path = r"G:\ML mostafa saad\slides\my work\2 Credit Card Fraud Detection\data\split\test.csv"
scaler_artifact_path = r"models/scaler.joblib"

# Load splits
df_train = load_dataset(train_path)
df_val = load_dataset(val_path)
df_test = load_dataset(test_path)

X_train, y_train = get_features_target(df_train)
X_val, y_val = get_features_target(df_val)
X_test, y_test = get_features_target(df_test)

# 1) Fit scaler and SMOTE on train
X_train_res, y_train_res, fitted_scaler = fit_scaler_and_transform_train(
    X_train=X_train,
    y_train=y_train,
    scaler_kind="standard",      # or "robust"
    apply_smote=True,
    smote_random_state=42,
    save_scaler_path=scaler_artifact_path,
    columns_to_scale=None        # None -> scale all columns
)

print("After SMOTE: X_train_res.shape =", X_train_res.shape, "y_train_res distribution:", np.bincount(y_train_res.astype(int)))

# 2) Transform val/test using saved scaler (no SMOTE!)
X_val_scaled = transform_with_saved_scaler(X_val, scaler_path=scaler_artifact_path)
X_test_scaled = transform_with_saved_scaler(X_test, scaler_path=scaler_artifact_path)

# 3) (Optional) Save processed arrays (useful for reproducibility)
save_processed_dataset(X_train_res, y_train_res, cols=None, out_csv_path="data/processed/train_processed.csv")
save_processed_dataset(X_val_scaled, y_val.values, cols=None, out_csv_path="data/processed/val_processed.csv")
save_processed_dataset(X_test_scaled, y_test.values, cols=None, out_csv_path="data/processed/test_processed.csv")
