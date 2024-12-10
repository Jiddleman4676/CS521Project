import time
import numpy as np
import pandas as pd
import random
from FNN import FeedforwardNeuralNetwork
from utils import sigmoid, NNN_loss, balanced_train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# for feature extraction
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Model comparison
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Hyperparameters (to make the program run faster, change training_size to 0.99 or EPOCHS to 10)
training_size = .1
LEARNING_RATE = 1.0
EPOCHS = 30
layer_sizes = [21, 128, 128, 3]

# Load CSV into a DataFrame
df = pd.read_csv('../data/diabetes_012_health_indicators_BRFSS2015.csv')

# PCA analysis
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
pca = PCA(n_components=.85)
principal_components = pca.fit_transform(scaled_data)
num_components = principal_components.shape[1]
print("num_components: ", num_components)
# end of PCA analysis

data = df.to_numpy()
y = data[:, 0]
y = y.astype(int)
X = data[:, 1:]

# print the number of samples per class in the entire dataset
unique_classes, counts = np.unique(y, return_counts=True)
print("")
print("number of samples per class in the entire dataset")
for cls, count in zip(unique_classes, counts):
    print(f"Class {cls}: {count} samples")
print("")

# Normalization
data_min = np.min(X, axis=0)
data_max = np.max(X, axis=0)
X = (X - data_min) / (data_max - data_min)

# Split data into train partition and test partition such that there are roughly equal numbers of samples per class (or until 80% of the samples in that class have been put into training data)
X_train, X_test, y_train, y_test = balanced_train_test_split(X, y, random_state=0, test_size=(1-training_size))

# Print the number of samples per class in the training data subsample
unique_classes, counts = np.unique(y_train, return_counts=True)
print("number of samples per class in the training data subsample")
for cls, count in zip(unique_classes, counts):
    print(f"Class {cls}: {count} samples")



print("training")
print(f"Layers: {layer_sizes[1:-1]} N: {len(X_train)}, LR: {LEARNING_RATE}, E: {EPOCHS}")

# train the neural network
start_time = time.time()

nn = FeedforwardNeuralNetwork(layer_sizes=layer_sizes, activations=[sigmoid, sigmoid,sigmoid])
nn.train(X_train, y_train, loss_function=NNN_loss,
         loss_derivative=lambda y_pred, y_true: NNN_loss(y_pred, y_true, derivative=True),
         epochs=EPOCHS, learning_rate=LEARNING_RATE)

end_time = time.time()
training_duration = np.round((end_time - start_time) / 60)  # Calculate duration

# Evaluate the NN model on train and test data (used for plots 1, 2, and 3)
# Predict the class of the train and test groups using the trained neural network model (first iteration is evaluation on the train, and the second is evaluation on the test subsample)

X_train_test = X_train
y_train_test = y_train

correct_counts_train = np.zeros(3)
incorrect_counts_train = np.zeros(3)
correct_counts_test = np.zeros(3)
incorrect_counts_test = np.zeros(3)

y_pred = np.zeros(len(y_train_test))

for i in range(2):
    
    if i == 1:
        X_train_test = X_test
        y_train_test = y_test
        y_pred = np.zeros(len(y_train_test))
    
    correct_counts = np.zeros(3)
    incorrect_counts = np.zeros(3)
    
    for j in range(len(y_train_test)):
        y_pred_all_probs = nn.forward(X_train_test[j])
        y_pred[j] = np.argmax(y_pred_all_probs)
        if y_pred[j] == y_train_test[j]:
            correct_counts[y_train_test[j]] += 1
        else:
            incorrect_counts[y_train_test[j]] += 1
    if i == 0:
        correct_counts_train = correct_counts
        incorrect_counts_train = incorrect_counts
    else:
        correct_counts_test = correct_counts
        incorrect_counts_test = incorrect_counts
print("")
print("correct_counts_train: ", correct_counts_train)
print("incorrect_counts_train: ", incorrect_counts_train)
print("correct_counts_test: ", correct_counts_test)
print("incorrect_counts_test: ", incorrect_counts_test)
print("")

# Model comparison for test dataset (used for plot 1)

# Neural network
accuracy_nn = np.sum(correct_counts_test) / (np.sum(correct_counts_test) + np.sum(incorrect_counts_test))
print("Neural Network")
print(f"Accuracy: {accuracy_nn:.2f}")
print(classification_report(y_test, y_pred, zero_division=0))
print("")

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree")
print(f"Accuracy: {accuracy_dt:.2f}")
print(classification_report(y_test, y_pred_dt, zero_division=0))
print("")

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print("Gaussian Naive Bayes")
print(f"Accuracy: {accuracy_gnb:.2f}")
print(classification_report(y_test, y_pred_gnb, zero_division=0))
print("")

# Support Vector Machine
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Support Vector Machine (SVM)")
print(f"Accuracy: {accuracy_svm:.2f}")
print(classification_report(y_test, y_pred_svm, zero_division=0))
print("")

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

# Plots #2 & #3: Neural network class statistics: train and test
X_train_test = X_train
y_train_test = y_train
correct_counts = correct_counts_train
incorrect_counts = incorrect_counts_train

for i in range(2):
    
    if i == 1:
        X_train_test = X_test
        y_train_test = y_test
        correct_counts = correct_counts_test
        incorrect_counts = incorrect_counts_test
    
    plt.figure(figsize=(3, 6))
    plt.bar(diabetes_status, correct_counts, label="Correct", color="blue")
    plt.bar(diabetes_status, incorrect_counts, bottom=correct_counts, label="Incorrect", color="red")

    plt.xlabel("Diabetic Status")
    plt.ylabel("Number of Classifications")
    plt.title(f"Correct vs Incorrect | Layers: {layer_sizes[1:-1]} N: {len(X_train_test)}, LR: {LEARNING_RATE}, E: {EPOCHS}, T:{training_duration} min,Correct: {((np.sum(correct_counts)/len(y_train_test)) * 100):.2f}%")
    status_names = ["Unafflicted", "Prediabetic", "Diabetic"]
    plt.xticks(diabetes_status, status_names)
    plt.legend()

    plt.show()
