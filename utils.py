import numpy as np
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import keras.backend as K
import tensorflow as tf


def get_auc(x, y):
    return roc_auc_score(np.reshape(y, (-1, )), np.reshape(x, (-1, )))


def get_err_rate(x, y):
    return np.sum(np.abs(x - y)) / np.sum(y)


def load_data(filePath):
    if not os.path.exists(filePath):
        raise FileNotFoundError
    else:
        return np.load(filePath)


def build_refined_loss(beta):

    def refined_loss(y_true, y_pred):
        weight = y_true * (beta - 1) + 1
        return K.mean(K.sum(tf.multiply(weight, K.square(y_true - y_pred)), axis=1), axis=-1)
    return refined_loss
