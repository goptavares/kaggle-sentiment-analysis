#!/usr/bin/python

"""
decision_trees.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import matplotlib.pyplot as plt
import operator

from sklearn import tree
from sklearn.cross_validation import cross_val_score

import util


X_train, Y_train, X_test = util.loadData()

minSamplesLeafRange = range(1, 20)
scoreCrossVal = list()
for minSamplesLeaf in minSamplesLeafRange:
    print("Running model " + str(minSamplesLeaf) + "...")
    clf = tree.DecisionTreeClassifier(criterion="gini",
                                      min_samples_leaf=minSamplesLeaf)
    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append(scores.mean())
index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimMinSamplesLeaf = minSamplesLeafRange[index]
print("Optimal minimum samples in leaf: " + str(optimMinSamplesLeaf))

plt.figure()
plt.plot(minSamplesLeafRange, scoreCrossVal)
plt.xlabel('Minimum samples in leaf node')
plt.ylabel('Cross validation score')
plt.title('Decision Tree')
plt.show()

maxDepthRange = range(5, 100, 5)
scoreCrossVal = list()
for maxDepth in maxDepthRange:
    print("Running model " + str(maxDepth) + "...")
    clf = tree.DecisionTreeClassifier(criterion="gini",
                                      max_depth=maxDepth)
    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append(scores.mean())
index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimMaxDepth = maxDepthRange[index]
print("Optimal max depth: " + str(optimMaxDepth))

plt.figure()
plt.plot(maxDepthRange, scoreCrossVal)
plt.xlabel('Maximum tree depth')
plt.ylabel('Cross validation score')
plt.title('Decision Tree')
plt.show()
