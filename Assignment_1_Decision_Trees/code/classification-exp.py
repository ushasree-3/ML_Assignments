import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

X = pd.DataFrame(X)
y = pd.Series(y)

print(X)
print(y)

# Write the code for Q2 a) and b) below. Show your results.
#a)
X_train, y_train = X[:70], y[:70]
X_test, y_test = X[70:].reset_index(drop = True), y[70:].reset_index(drop = True)

tree = DecisionTree(criterion="information_gain")  # Split based on Inf. Gain
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)

print(y_hat)

print("Accuracy of Decision tree: ", accuracy(y_hat, y_test))

cls = np.unique(y)
print("per-class precision")
for i in cls:
  print("cls:", i, "; precision:", precision(y_hat, y_test, i))

#taking cls = 1
print("Recall of Decision tree: ", recall(y_hat, y_test, 1))

#b)

def k_fold_cross_validation(X, y, k = 5):
  fold_size = len(X) // k
  accuracies = []

  for i in range(k):
    test_start = i * fold_size
    test_end = (i + 1) * fold_size
    test_set = X[test_start:test_end]
    test_labels = y[test_start:test_end]

    training_set = pd.DataFrame(np.concatenate((X[:test_start], X[test_end:]), axis=0))
    training_labels = pd.Series(np.concatenate((y[:test_start], y[test_end:]), axis=0))

    clf = DecisionTree(criterion = "information_gain")
    clf.fit(training_set, training_labels)

    fold_predictions = clf.predict(test_set)
    fold_accuracy = np.mean(fold_predictions == test_labels)

    accuracies.append(fold_accuracy)
    return accuracies

def nested_cross_validation(X, y, depth_start, depth_end, k_outer_folds = 5, k_inner_folds = 5): #depth end included
  optimal_depth = 0
  highest_accuracy = 0

  for i in range(k_outer_folds):
    test_start = i * k_outer_folds
    test_end = (i + 1) * k_outer_folds
    test_set = X[test_start:test_end]
    test_labels = y[test_start:test_end]

    training_set = np.concatenate((X[:test_start], X[test_end:]), axis=0)
    training_labels = np.concatenate((y[:test_start], y[test_end:]), axis=0)

    accuracy_per_depth = []
    for d in range(depth_start, depth_end+1):
      accuracies = []

      for j in range(k_inner_folds):
        train_val_start = j * k_inner_folds
        train_val_end = (j + 1) * k_inner_folds
        train_val_set = pd.DataFrame(training_set[train_val_start:train_val_end])
        train_val_labels = pd.Series(training_labels[train_val_start:train_val_end])

        training_set_new = pd.DataFrame(np.concatenate((training_set[:train_val_start], training_set[train_val_end:]), axis=0))
        training_labels_new = pd.Series(np.concatenate((training_labels[:train_val_start], training_labels[train_val_end:]), axis=0))

        clf = DecisionTree( criterion = "",max_depth = d)  #-----------------
        clf.fit(training_set_new, training_labels_new)
        y_hat = clf.predict(train_val_set)

        accuracies.append(accuracy(y_hat, train_val_labels))

      accuracy_per_depth.append(np.mean(accuracies))

    optimal_depth_outerfold = depth_start + np.argmax(accuracy_per_depth)

    if np.mean(accuracy_per_depth) > highest_accuracy:
      highest_accuracy = np.mean(accuracy_per_depth)
      optimal_depth = optimal_depth_outerfold

  return optimal_depth

nested_cross_validation(X, y, 1, 10)
