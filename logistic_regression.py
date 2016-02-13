#!/usr/bin/python

"""
logistic_regression.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import matplotlib.pyplot as plt
import numpy as np
import operator

from sklearn import linear_model
from sklearn.cross_validation import cross_val_score

import util


X_train, Y_train, X_test = util.loadData(normalize=True)

# L1 regularization.
regularizationRange = np.arange(0.01, 0.4, 0.01)
scoreCrossVal = list()
for reg in regularizationRange:
    print("Running model " + str(reg) + "...")
    clf = linear_model.LogisticRegression(penalty='l1', C=reg, dual=False)
    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append(scores.mean())

index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimReg = regularizationRange[index]
print("Optimal regularization parameter: " + str(optimReg))
l1_scores = scoreCrossVal

# L2 regularization.
regularizationRange = np.arange(0.01, 0.4, 0.01)
scoreCrossVal = list()
for reg in regularizationRange:
    print("Running model " + str(reg) + "...")
    clf = linear_model.LogisticRegression(penalty='l2', C=reg, dual=False)
    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append(scores.mean())

index, val =  max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimReg = regularizationRange[index]
print("Optimal regularization parameter: " + str(optimReg))
l2_scores = scoreCrossVal

plt.figure()
plt.plot(regularizationRange, l1_scores)
plt.plot(regularizationRange, l2_scores)
plt.xlabel('Weight of penalty term')
plt.ylabel('Cross validation score')
plt.title('Logistic Regression')
plt.legend(['L1', 'L2'], loc='lower right')
plt.show()