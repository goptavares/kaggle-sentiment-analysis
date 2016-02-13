#!/usr/bin/python

"""
linear_classifiers.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np

from sklearn import linear_model
from sklearn.cross_validation import cross_val_score

import util


X_train, Y_train, X_test = util.loadData(normalize=True)

# Lasso.
clf = linear_model.Lasso(alpha=0.001)
scores = cross_val_score(clf, X_train, Y_train)
print("Lasso cross validation mean: " + str(scores.mean()))

clf.fit(X_train, Y_train)
util.writeData(clf.predict(X_test))

# Ridge.
clf = linear_model.Ridge(alpha=0.001)
scores = cross_val_score(clf, X_train, Y_train) 
print("Ridge cross validation mean: " + str(scores.mean()))

clf.fit(X_train, Y_train)
util.writeData(clf.predict(X_test))
