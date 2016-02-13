#!/usr/bin/python

"""
discriminant_analysis.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import operator

from sklearn.cross_validation import cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import util


X_train, Y_train, X_test = util.loadData(normalize=True)

regParamRange = np.arange(0.001, 0.3, 0.001)
scoreCrossVal = list()
for regParam in regParamRange:
    print("Running model " + str(regParam) + "...")
    clf = QuadraticDiscriminantAnalysis(reg_param=regParam)
    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append(scores.mean())
index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimRegParam = regParamRange[index]
print("Optimal regularization parameter: " + str(optimRegParam))
