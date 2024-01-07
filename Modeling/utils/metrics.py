import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score,ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def precision(y_true, y_pred):
    """Precision metric.
    
    Only computes a batch-wise average of precision.
    
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred,0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred,0,1)))
    prec = true_positives / (predicted_positives + K.epsilon())
    return prec


def recall(y_true, y_pred):
    """ Recall metric.
    
    Only computes a batch-wise average of recall.
    
    Computes the recall, a metric for multi-label classification of 
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
    posible_positives = K.sum(K.round(K.clip(y_true,0,1)))
    rec = true_positives / (posible_positives + K.epsilon())
    return rec


def f1(y_true, y_pred):

    """ F1 metric.

    Only computes a batch-wise average of F1.

    Computes the F1, a metric for multi-label classification of
    how many selected items are relevant and how many relevant items are selected.
    """
    prec= precision(y_true,y_pred)
    rec = recall(y_true,y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))



def confusion_matrix_plot(y_true, y_pred):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    confusion_matrices = confusion_matrix(y_true, y_pred,labels=np.unique(y_pred), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrices, display_labels=np.unique(y_pred))
    return disp