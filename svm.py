#!/usr/bin/python

"""
svm.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import matplotlib.pyplot as plt
import numpy as np
import operator

from sklearn import svm
from sklearn.cross_validation import cross_val_score

import util


X_train, Y_train, X_test = util.loadData(normalize=True)

# SVC with RBF kernel.
# Perform cross validation to find the optimal penalty parameter, in order to
# avoid overfitting.
penaltyRange = np.arange(1.5, 3, 0.1)
scoreCrossVal = list()
for penalty in penaltyRange:
    print("Running model " + str(penalty) + "...")
    clf = svm.SVC(C=penalty, kernel='rbf')
    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append( scores.mean())

index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimPenalty = penaltyRange[index]
print("Optimal penalty parameter: " + str(optimPenalty))

plt.figure()
plt.plot(penaltyRange, scoreCrossVal)
plt.xlabel('Penalty parameter of error term')
plt.ylabel('Cross validation score')
plt.title('SVM')
plt.show()
