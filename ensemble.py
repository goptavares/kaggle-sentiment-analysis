#!/usr/bin/python

"""
ensemble.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB

import util

X_train, Y_train, X_test = util.loadData()

# SVM.
clf1 = svm.SVC(probability=True, C=2.915)
clf1.fit(X_train, Y_train)

# AdaBoost.
clf2 = AdaBoostClassifier(n_estimators=170, learning_rate=0.85)
clf2.fit(X_train, Y_train)

# Random forest with min leaf node size.
clf3 = RandomForestClassifier(n_estimators=100, criterion='gini',
						      min_samples_leaf=2)
clf3.fit(X_train, Y_train)

# Random forest with max tree depth.
clf4 = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=62)
clf4.fit(X_train, Y_train)

# Create ensemble using weighted average probabilities (soft voting) and
# predict on the test dataset.
eclf = VotingClassifier(estimators=[('svm', clf1), ('adaboost', clf2),
                        ('rf1', clf3), ('rf2', clf4)], voting='soft')
eclf.fit(X_train,Y_train)
util.writeData(eclf.predict(X_test))