#!/usr/bin/python

"""
gradient_boosting.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import operator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

import util


X_train, Y_train, X_test = util.loadData()

# Gradient boosting.
# Perform cross validation to find the optimal number of estimators and max tree
# depth, in order to avoid overfitting.
numEstimatorsRange = range(105, 120, 5)
maxDepthRange = range(4, 13, 3)
errorTrain = list()
errorCrossVal = list()
models = list()
for numEstimators in numEstimatorsRange:
    for maxDepth in maxDepthRange:
        models.append((numEstimators, maxDepth))
        print("Running model: " + str((numEstimators, maxDepth)) + "...")
        clf = GradientBoostingClassifier(n_estimators=numEstimators,
                                         max_depth=maxDepth)
        scores = cross_val_score(clf, X_train, Y_train)
        errorCrossVal.append((1 - scores.mean()) * 100)

        clf.fit(X_train, Y_train)
        errorTrain.append((1 - clf.score(X_train, Y_train)) * 100)

minIndex, minValue = min(enumerate(errorCrossVal), key=operator.itemgetter(1))
print("Min cross validation error: " + str(minValue))
optimParams = models[minIndex]
print("Optimal number of estimators and max tree depth: " + str(optimParams))

# Use the optimal classifier to predict on the test dataset.
clf = GradientBoostingClassifier(n_estimators=optimParams[0],
                                 max_depth=optimParams[1])
clf.fit(X_train, Y_train)
util.writeData(clf.predict(X_test))

# Optimal number of estimators: 90 from range(10, 110, 20)
# Optimal max tree depth: 12 from range(2, 102, 10)

# Optimal number of estimators: 110 from range(80, 130, 10)
# Optimal max tree depth: 7 from range(2, 22, 5)

# Optimal number of estimators: 115 from range(105, 120, 5)
# Optimal max tree depth: 7 from range(4, 13, 3)