#!/usr/bin/python

"""
adaboost.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import operator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score

import util


X_train, Y_train, X_test = util.loadData()

# AdaBoost.
# Perform cross validation to find the optimal number of estimators and learning
# rate, in order to avoid overfitting.
numEstimatorsRange = range(10, 210, 20)
learningRateRange = np.arange(0.05, 1., 0.1)
errorTrain = list()
errorCrossVal = list()
models = list()
for numEstimators in numEstimatorsRange:
    for learningRate in learningRateRange:
        models.append((numEstimators, learningRate))
        clf = AdaBoostClassifier(n_estimators=numEstimators,
                                 learning_rate=learningRate)
        scores = cross_val_score(clf, X_train, Y_train)
        errorCrossVal.append((1 - scores.mean()) * 100)

        clf = clf.fit(X_train, Y_train)
        errorTrain.append((1 - clf.score(X_train, Y_train)) * 100)

minIndex, minValue = min(enumerate(errorCrossVal), key=operator.itemgetter(1))
print("Min cross validation error: " + str(minValue))
optimParams = models[minIndex]
print("Optimal number of estimators and learning rate: " + str(optimParams))

# Use the optimal classifier to predict on the test dataset.
clf = AdaBoostClassifier(n_estimators=optimParams[0],
                         learning_rate=optimParams[1])
clf = clf.fit(X_train, Y_train)
util.writeData(clf.predict(X_test))

# Optimal number of estimators: 170
# Optimal learning rate: 0.85
