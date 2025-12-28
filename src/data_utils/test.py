import sys
import os

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_utils.loader import load_dataset, get_features_target
from src.data_utils.preprocessing import fit_scaler_and_transform_train

save_path = r"G:\ML mostafa saad\slides\my work\2 Credit Card Fraud Detection\models\scaler.joblib"

df_train = load_dataset(r"G:\ML mostafa saad\slides\my work\2 Credit Card Fraud Detection\data\split\train.csv")
X_train, y_train = get_features_target(df_train)

X_train_res, y_train_res, scaler = fit_scaler_and_transform_train(
    X_train=X_train,
    y_train=y_train,
    scaler_kind="standard",
    apply_smote=True,
    save_scaler_path=save_path
)

print(f"\n[CHECK] Does scaler file exist? {os.path.exists(save_path)}")
print(f"[CHECK] Absolute path: {os.path.abspath(save_path)}")
