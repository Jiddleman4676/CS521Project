import ssl
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from FNN import FeedforwardNeuralNetwork
from utils import sigmoid, NNN_loss
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import random

LEARNING_RATE = 1
EPOCHS = 10
layer_sizes = [21, 128, 128, 3]

# Load CSV into a DataFrame
df = pd.read_csv('../data/diabetes_012_health_indicators_BRFSS2015.csv')
data = df.to_numpy()
y = data[:, 0]  # First column
X = data[:, 1:]  # All other columns

#print("X shape: ", X.shape)
#print("y shape: ", y.shape)

# Iterate through each column in X and print column name and unique values
for idx, column_name in enumerate(df.columns[1:]):  # Start from the second column (X)
    unique_values = np.unique(X[:, idx])  # Get unique values for the column
    print(f"Column: {column_name}")
    print(f"Unique values: {unique_values}")
    print("-" * 30)  # Separator for readability

# Normalization (from features with varying ranges to a set range of [0,1] for all features.
data_min = np.min(X, axis=0)  # Minimum value along each feature
data_max = np.max(X, axis=0)  # Maximum value along each feature
X = (X - data_min) / (data_max - data_min)

# Iterate through each column in X and print column name and unique values
for idx, column_name in enumerate(df.columns[1:]):  # Start from the second column (X)
    unique_values = np.unique(X[:, idx])  # Get unique values for the column
    print(f"Column: {column_name}")
    print(f"Unique values: {unique_values}")
    print("-" * 30)  # Separator for readability

## Split data into train partition and test partition
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.95)
#_, X_test, _, y_test = train_test_split(X_test, y_test, random_state=0, test_size=0.1)
## Define the FNN model with 784 input neuron, 64 hidden neurons in 2 layers, and 10 output neuron
#nn = FeedforwardNeuralNetwork(layer_sizes=layer_sizes, activations=[sigmoid, sigmoid,sigmoid,sigmoid])
#
#print("training")
#print(f"Layers: {layer_sizes[1:-1]} N: {len(X_train)}, LR: {LEARNING_RATE}, E: {EPOCHS}")
#start_time = time.time()  # Record start time
#
#nn.train(X_train, y_train, loss_function=NNN_loss,
#         loss_derivative=lambda y_pred, y_true: NNN_loss(y_pred, y_true, derivative=True),
#         epochs=EPOCHS, learning_rate=LEARNING_RATE)
#
#end_time = time.time()  # Record end time
#training_duration = np.round((end_time - start_time) / 60)  # Calculate duration
#
#print("test")
#
## Evaluate model on test data
#correct_counts = np.zeros(3)
#incorrect_counts = np.zeros(3)
#
#for i in range(len(y_test)):
#    y_pred = nn.forward(X_test[i])
#    if np.random.randint(1, 1000) == 1:
#        print(y_pred)
#    if np.argmax(y_pred)  ==  y_test[i]:
#        correct_counts[y_test[i]] += 1
#    else:
#        incorrect_counts[y_test[i]] += 1
#
#print("correct: ", np.sum(correct_counts), "incorrect: ", np.sum(incorrect_counts))
#print("accuracy: ", np.sum(correct_counts) / (np.sum(correct_counts) + np.sum(incorrect_counts))) 
#
#covtype = np.arange(3)
#
## Plotting correct vs incorrect classifications for each digit
#plt.figure(figsize=(3, 6)) #??
#plt.bar(covtype, correct_counts, label="Correct", color="blue")
#plt.bar(covtype, incorrect_counts, bottom=correct_counts, label="Incorrect", color="red")
#
## Labeling the plot
#plt.xlabel("Diabetic Status")
#plt.ylabel("Number of Classifications")
#plt.title(f"Correct vs Incorrect | Layers: {layer_sizes[1:-1]} N: {len(X_train)}, LR: {LEARNING_RATE}, E: {EPOCHS}, T:{training_duration} min,Correct: {((np.sum(correct_counts)/len(y_test)) * 100):.2f}%")
#plt.xticks(covtype)
#plt.legend()
#
## Show plot
#plt.show()
