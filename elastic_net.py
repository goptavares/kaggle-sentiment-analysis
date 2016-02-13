#!/usr/bin/python

"""
elastic_net.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import operator

from sklearn import linear_model
from sklearn.cross_validation import cross_val_score

import util

X_train, Y_train, X_test = util.loadData()

# L1 regularization.
alphaRange = np.arange(0.01, 0.1, 0.01)
l1RatioRange = np.arange(0.1, 0.5, 0.1)
scoreCrossVal = list()
models = list()
for alpha in alphaRange:
    for l1Ratio in l1RatioRange:
        models.append((alpha, l1Ratio))
        print("Running model: " + str((alpha, l1Ratio)) + "...")
        clf = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1Ratio)
        scores = cross_val_score(clf, X_train, Y_train)
        print(scores)
        scoreCrossVal.append(scores.mean())

maxIndex, maxValue = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(maxValue))
optimParams = models[maxIndex]
print("Optimal alpha and L1 ratio: " + str(optimParams))

# Use the optimal classifier to predict on the test dataset.
clf = linear_model.ElasticNet(alpha=optimParams[0], l1_ratio=optimParams[1])
clf.fit(X_train, Y_train)
util.writeData(clf.predict(X_test))
