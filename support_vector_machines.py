#!/usr/bin/python

"""
support_vector_machines.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
from sklearn import svm
from sklearn.cross_validation import cross_val_score

import util

X_train, Y_train, X_test = util.loadData()
clf = svm.SVC()

scores = cross_val_score(clf, X_train, Y_train) 
print("Cross validation mean: " + str(scores.mean()))

clf.fit(X_train, Y_train)
util.writeData(clf.predict(X_test))