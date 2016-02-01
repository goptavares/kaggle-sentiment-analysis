#!/usr/bin/python

"""
random_forests.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import operator
import util


# Load the training data.
X_train, Y_train, X_test = util.loadData()

# Perform cross validation to find the optimal min leaf node size, in order to
# avoid overfitting.
leafNodeSizeRange = range(1, 101)
errorTrain = list()
errorCrossVal = list()
for minLeafNodeSize in leafNodeSizeRange:
    clf = RandomForestClassifier(n_estimators=100, criterion='gini',
                                 min_samples_leaf=minLeafNodeSize)

    scores = cross_val_score(clf, X_train, Y_train)
    errorCrossVal.append((1 - scores.mean()) * 100)

    clf = clf.fit(X_train, Y_train)
    errorTrain.append((1 - clf.score(X_train, Y_train)) * 100)

minIndex, minValue = min(enumerate(errorCrossVal), key=operator.itemgetter(1))
print("Min cross validation error: " + str(minValue))
optimLeafNodeSize = leafNodeSizeRange[minIndex]
print("Optimal min leaf node size: " + str(optimLeafNodeSize))

plt.figure()
plt.plot(leafNodeSizeRange, errorTrain)
plt.plot(leafNodeSizeRange, errorCrossVal)
plt.xlabel('Min leaf node size')
plt.ylabel('Error %')
plt.legend(['Training', 'Cross validation mean'], loc='lower right')
plt.show()

# Use the optimal classifier to predict on the test dataset.
clf = RandomForestClassifier(n_estimators=100, criterion='gini',
                             min_samples_leaf=optimLeafNodeSize)
clf = clf.fit(X_train, Y_train)
util.writeData(clf.predict(X_test))

# Perform cross validation to find the optimal max tree depth, in order to
# avoid overfitting.
maxDepthRange = range(2, 101)
errorTrain = list()
errorCrossVal = list()
for maxTreeDepth in maxDepthRange:
    clf = RandomForestClassifier(n_estimators=100, criterion='gini',
                                 max_depth=maxTreeDepth)

    scores = cross_val_score(clf, X_train, Y_train)
    errorCrossVal.append((1 - scores.mean()) * 100)

    clf = clf.fit(X_train, Y_train)
    errorTrain.append((1 - clf.score(X_train, Y_train)) * 100)

minIndex, minValue = min(enumerate(errorCrossVal), key=operator.itemgetter(1))
print("Min cross validation error: " + str(minValue))
optimTreeDepth = maxDepthRange[minIndex]
print("Optimal max tree depth: " + str(optimTreeDepth))

plt.figure()
plt.plot(maxDepthRange, errorTrain)
plt.plot(maxDepthRange, errorCrossVal)
plt.xlabel('Max tree depth')
plt.ylabel('Error %')
plt.legend(['Training', 'Cross validation mean'], loc='lower right')
plt.show()

# Use the optimal classifier to predict on the test dataset.
clf = RandomForestClassifier(n_estimators=100, criterion='gini',
                             max_depth=optimTreeDepth)
clf = clf.fit(X_train, Y_train)
util.writeData(clf.predict(X_test))

# Optimal min leaf node size: 2
# Optimal max tree depth: 62
