import time
import numpy as np
import pandas as pd
import random
from FNN import FeedforwardNeuralNetwork
from utils import sigmoid, NNN_loss, balanced_train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#Model comparison
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import make_classification

LEARNING_RATE = 1.0
EPOCHS = 1
layer_sizes = [21, 128, 128, 3]

# Load CSV into a DataFrame
df = pd.read_csv('../data/diabetes_012_health_indicators_BRFSS2015.csv')
data = df.to_numpy()
y = data[:, 0]  # First column
y = y.astype(int)

# Assume y is a NumPy array of class labels
unique_classes, counts = np.unique(y, return_counts=True)

# Print the count for each class
for cls, count in zip(unique_classes, counts):
    print(f"Class {cls}: {count} samples")

X = data[:, 1:]  # All other columns

# Normalization (from features with varying ranges to a set range of [0,1] for all features.
data_min = np.min(X, axis=0)  # Minimum value along each feature
data_max = np.max(X, axis=0)  # Maximum value along each feature
X = (X - data_min) / (data_max - data_min)

# Split data into train partition and test partition
#X_train, X_test, y_train, y_test = balanced_train_test_split(X, y, random_state=0, test_size=0.99)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.99)


# Assume y is a NumPy array of class labels
unique_classes, counts = np.unique(y_train, return_counts=True)

# Print the count for each class
for cls, count in zip(unique_classes, counts):
    print(f"Class {cls}: {count} samples")




_, X_test, _, y_test = train_test_split(X_test, y_test, random_state=0, test_size=0.2, stratify=y_test)
# Define the FNN model with 784 input neuron, 64 hidden neurons in 2 layers, and 10 output neuron
nn = FeedforwardNeuralNetwork(layer_sizes=layer_sizes, activations=[sigmoid, sigmoid,sigmoid,sigmoid])

print("training")
print(f"Layers: {layer_sizes[1:-1]} N: {len(X_train)}, LR: {LEARNING_RATE}, E: {EPOCHS}")
start_time = time.time()

nn.train(X_train, y_train, loss_function=NNN_loss,
         loss_derivative=lambda y_pred, y_true: NNN_loss(y_pred, y_true, derivative=True),
         epochs=EPOCHS, learning_rate=LEARNING_RATE)

end_time = time.time()
training_duration = np.round((end_time - start_time) / 60)  # Calculate duration

# Evaluate model on test data

# Predicting the class of the test group using the neural network model

correct_counts = np.zeros(3)
incorrect_counts = np.zeros(3)

y_pred = np.zeros(len(y_test))

for i in range(len(y_test)):
    y_pred_all_probs = nn.forward(X_test[i])
    y_pred[i] = np.argmax(y_pred_all_probs)
    if np.random.randint(1, 1000) == 1:
        print(y_pred_all_probs)
    if y_pred[i] == y_test[i]:
        correct_counts[y_test[i]] += 1
    else:
        incorrect_counts[y_test[i]] += 1

# Model comparison

# Neural network
accuracy_nn = np.sum(correct_counts) / (np.sum(correct_counts) + np.sum(incorrect_counts))
print("Neural Network")
print(f"Accuracy: {accuracy_nn:.2f}")
print(classification_report(y_test, y_pred, zero_division=0))
print("-" * 50)

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree")
print(f"Accuracy: {accuracy_dt:.2f}")
print(classification_report(y_test, y_pred_dt, zero_division=0))
print("-" * 50)

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print("Gaussian Naive Bayes")
print(f"Accuracy: {accuracy_gnb:.2f}")
print(classification_report(y_test, y_pred_gnb, zero_division=0))
print("-" * 50)

# Support Vector Machine
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Support Vector Machine (SVM)")
print(f"Accuracy: {accuracy_svm:.2f}")
print(classification_report(y_test, y_pred_svm, zero_division=0))
print("-" * 50)

# Plot #1: Model comparison
model_names = ["Neural Network", "Decision Tree", "Gaussian Naive Bayes", "SVM"]
accuracies = [accuracy_nn, accuracy_dt, accuracy_gnb, accuracy_svm]

plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies, color=['red', 'blue', 'green', 'purple'])
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Comparison of Model Accuracies")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()
diabetes_status = np.arange(3)

# Plot #2: Neural network class statistics

# Plotting correct vs incorrect classifications for each digit
plt.figure(figsize=(3, 6))
plt.bar(diabetes_status, correct_counts, label="Correct", color="blue")
plt.bar(diabetes_status, incorrect_counts, bottom=correct_counts, label="Incorrect", color="red")

# Labeling the plot
plt.xlabel("Diabetic Status")
plt.ylabel("Number of Classifications")
plt.title(f"Correct vs Incorrect | Layers: {layer_sizes[1:-1]} N: {len(X_train)}, LR: {LEARNING_RATE}, E: {EPOCHS}, T:{training_duration} min,Correct: {((np.sum(correct_counts)/len(y_test)) * 100):.2f}%")
status_names = ["Unafflicted", "Prediabetic", "Diabetic"]
plt.xticks(diabetes_status, status_names)
plt.legend()

# Show plot
plt.show()
