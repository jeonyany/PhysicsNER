import numpy as np
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
    

def get_metrics_ops(labels, predictions, num_labels):
    cm, op = _streaming_confusion_matrix(labels, predictions, num_labels+1)
    tf.logging.info(type(cm))
    tf.logging.info(type(op))
    
    return (tf.convert_to_tensor(cm), op)


def get_metrics(conf_mat, num_labels):
    precisions = []
    recalls = []
    for i in range(num_labels):
        tp = conf_mat[i][i].sum()
        col_sum = conf_mat[:, i].sum()
        row_sum = conf_mat[i].sum()

        precision = tp / col_sum if col_sum > 0 else 0
        recall = tp / row_sum if row_sum > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    pre = sum(precisions) / len(precisions)
    rec = sum(recalls) / len(recalls)
    f1 = 2 * pre * rec / (pre + rec)

    return pre, rec, f1