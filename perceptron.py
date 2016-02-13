#!/usr/bin/python

"""
perceptron.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import operator

from sklearn import linear_model
from sklearn.cross_validation import cross_val_score

import util


X_train, Y_train, X_test = util.loadData(normalize=True)

# L1 regularization.
alphaRange = np.arange(0.0001, 0.1, 0.0001)
scoreCrossVal = list()
for alpha in alphaRange:
    print("Running model " + str(alpha) + "...")
    clf = linear_model.Perceptron(penalty='l1', alpha=alpha)
    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append(scores.mean())

index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Min cross validation error: " + str(val))
optimAlpha = alphaRange[index]
print("Optimal alpha: " + str(optimAlpha))

# L2 regularization.
alphaRange = np.arange(0.0001, 0.1, 0.0001)
scoreCrossVal = list()
for alpha in alphaRange:
    print("Running model " + str(alpha) + "...")
    clf = linear_model.Perceptron(penalty='l2', alpha=alpha)
    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append(scores.mean())

index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Min cross validation error: " + str(val))
optimAlpha = alphaRange[index]
print("Optimal alpha: " + str(optimAlpha))
