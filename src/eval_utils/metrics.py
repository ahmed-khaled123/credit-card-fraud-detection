# src/eval_utils/metrics.py

import os
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

RESULT_DIR = r"G:\ML mostafa saad\slides\my work\2 Credit Card Fraud Detection\results"


def generate_metrics_report(y_true, y_pred, y_prob, save_txt=True):
    """
    Generates full evaluation report:
    - F1, Precision, Recall, ROC-AUC, PR-AUC
    - Saves metrics_report.txt automatically
    """

    report_text = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    final_text = f"""
================= Model Evaluation Report =================

 Classification Report:
{report_text}

📊 Additional Metrics
----------------------
ROC-AUC Score   = {roc_auc:.4f}
PR-AUC Score    = {pr_auc:.4f}
F1 Score        = {f1_score(y_true, y_pred):.4f}

============================================================
"""

    print(final_text)

    if save_txt:
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(f"{RESULT_DIR}/metrics_report.txt", "w") as f:
            f.write(final_text)

        print(f" Report Saved → {RESULT_DIR}/metrics_report.txt")

    return {
        "classification_report": report_text,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def save_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    os.makedirs(RESULT_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{RESULT_DIR}/{filename}")
    plt.close()

    print(f" Saved → {RESULT_DIR}/{filename}")


def save_roc_curve(y_true, y_prob, filename="roc_curve.png"):
    os.makedirs(RESULT_DIR, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{RESULT_DIR}/{filename}")
    plt.close()

    print(f" Saved → {RESULT_DIR}/{filename}")