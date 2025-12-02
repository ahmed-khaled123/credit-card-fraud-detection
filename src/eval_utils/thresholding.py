import numpy as np
thresholds = np.arange(0.0, 1.0, 0.01)
print(thresholds)
# import numpy as np
# from sklearn.metrics import precision_recall_curve, f1_score

# def find_best_threshold(y_true, y_prob, metric="f1"):
#     """
#     Search for best threshold (0 → 1) based on chosen metric.
#     metric options → "f1", "precision", "recall".
#     """

#     thresholds = np.arange(0.0, 1.0, 0.01)
#     print(thresholds)
#     best_t, best_score = 0, -1

#     for t in thresholds:
#         y_pred = (y_prob >= t).astype(int)

#         if metric == "precision":
#             from sklearn.metrics import precision_score
#             score = precision_score(y_true, y_pred, zero_division=0)

#         elif metric == "recall":
#             from sklearn.metrics import recall_score
#             score = recall_score(y_true, y_pred, zero_division=0)

#         else:  # default F1
#             score = f1_score(y_true, y_pred, zero_division=0)

#         if score > best_score:
#             best_score, best_t = score, t

#     return best_t, best_score


# def optimize_threshold_all(y_true, y_prob):
#     """
#     Returns best threshold based on F1, Precision, Recall.
#     """

#     results = {}
#     for m in ["f1", "precision", "recall"]:
#         t, s = find_best_threshold(y_true, y_prob, metric=m)
#         results[m] = {"threshold": t, "score": round(s, 4)}

#     return results
