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
