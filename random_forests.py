#!/usr/bin/python

"""
random_forests.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import matplotlib.pyplot as plt
import operator

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import util


# Load the training data.
X_train, Y_train, X_test = util.loadData(normalize=True)

# Perform cross validation to find the optimal min leaf node size, in order to
# avoid overfitting.
leafNodeSizeRange = range(1, 11)
scoreCrossVal = list()
for minLeafNodeSize in leafNodeSizeRange:
    clf = RandomForestClassifier(n_estimators=300, criterion='gini',
                                 min_samples_leaf=minLeafNodeSize)
    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append(scores.mean())

index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimLeafNodeSize = leafNodeSizeRange[index]
print("Optimal min leaf node size: " + str(optimLeafNodeSize))

plt.figure()
plt.plot(leafNodeSizeRange, scoreCrossVal)
plt.xlabel('Minimum samples in leaf node')
plt.ylabel('Cross validation score')
plt.title('Random Forest')
plt.show()

# Perform cross validation to find the optimal max tree depth, in order to
# avoid overfitting.
maxDepthRange = range(30, 100, 5)
scoreCrossVal = list()
for maxTreeDepth in maxDepthRange:
    clf = RandomForestClassifier(n_estimators=300, criterion='gini',
                                 max_depth=maxTreeDepth)

    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append(scores.mean())

index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimTreeDepth = maxDepthRange[index]
print("Optimal max tree depth: " + str(optimTreeDepth))

plt.figure()
plt.plot(maxDepthRange, scoreCrossVal)
plt.xlabel('Maximum tree depth')
plt.ylabel('Cross validation score')
plt.title('Random Forest')
plt.show()

# Try an extremely randomized forest.
leafNodeSizeRange = range(1, 11)
scoreCrossVal = list()
for minLeafNodeSize in leafNodeSizeRange:
    print("Running model " + str(minLeafNodeSize) + "...")
    clf = ExtraTreesClassifier(n_estimators=400, criterion='gini',
                               min_samples_leaf=minLeafNodeSize)
    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append(scores.mean())

index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimLeafNodeSize = leafNodeSizeRange[index]
print("Optimal min leaf node size: " + str(optimLeafNodeSize))

plt.figure()
plt.plot(leafNodeSizeRange, scoreCrossVal)
plt.xlabel('Minimum samples in leaf node')
plt.ylabel('Cross validation score')
plt.title('Extremely Randomized Forest')
plt.show()
