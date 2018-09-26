import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow.keras.backend as K

def my_f1(y_true, y_pred):
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    all_pos = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if all_pos == 0 or pred_pos == 0:
        return 0

    precision = true_pos / pred_pos
    recall = true_pos / all_pos
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score