#!/usr/bin/python

"""
linear_svm.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import operator
from sklearn import svm
from sklearn.cross_validation import cross_val_score

import util


X_train, Y_train, X_test = util.loadData()

# Linear SVC.
# Perform cross validation to find the optimal penalty parameter, in order to
# avoid overfitting.
penaltyRange = np.arange(0.01, 1., 0.02)
errorTrain = list()
errorCrossVal = list()
for penalty in penaltyRange:
	clf = svm.LinearSVC(C=penalty, penalty='l1', loss='squared_hinge',
					    dual=False)

	scores = cross_val_score(clf, X_train, Y_train)
	errorCrossVal.append((1 - scores.mean()) * 100)

	clf = clf.fit(X_train, Y_train)
	errorTrain.append((1 - clf.score(X_train, Y_train)) * 100)

minIndex, minValue = min(enumerate(errorCrossVal), key=operator.itemgetter(1))
print("Min cross validation error: " + str(minValue))
optimPenalty = penaltyRange[minIndex]
print("Optimal penalty parameter: " + str(optimPenalty))

# Use the optimal classifier to predict on the test dataset.
clf = svm.SVC(C=optimPenalty)
clf = clf.fit(X_train, Y_train)
util.writeData(clf.predict(X_test))

# Optimal penalty parameter for Linear SVC: 0.007 (for L2 penalty)
# Optimal penalty parameter for Linear SVC: 0.19 (for L1 penalty)
