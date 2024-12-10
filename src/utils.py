import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from FNN import FeedforwardNeuralNetwork
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
import random

# Store the activation functions along with their derivatives
def sigmoid(x, derivative=False):
    x = np.clip(x, -100, 100) # Prevent weights from overflowing
    x += + 1e-10 # Prevent divide by 0
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

# Custom loss function with optional derivative
def NNN_loss(y_pred, y_true, derivative=False):
    if derivative:
        grad = np.copy(y_pred)
        grad[range(len(y_pred)), y_true] -= 1
        return grad / len(y_pred)  # Gradient with respect to the input

    return np.mean(-y_pred[range(len(y_pred)), y_true])

def balanced_train_test_split(X, y, test_size, random_state=None):
    test_size = 1 - test_size
    class_size = int(test_size * (1/3)*len(y))
    classes = np.unique(y)
    X_train, X_test = [], []
    y_train, y_test = [], []

    for cls in classes:
        # Get indices for the current class
        cls_indices = np.where(y == cls)[0]
        cls_X = X[cls_indices]
        cls_y = y[cls_indices]

        # Perform train_test_split for the current class
        X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
                cls_X, cls_y, train_size=class_size, random_state=random_state
        )

        # Append the results for this class
        X_train.append(X_cls_train)
        X_test.append(X_cls_test)
        y_train.append(y_cls_train)
        y_test.append(y_cls_test)

    # Concatenate all classes
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.hstack(y_train)
    y_test = np.hstack(y_test)

    return X_train, X_test, y_train, y_test

