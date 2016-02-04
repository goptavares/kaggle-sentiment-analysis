#!/usr/bin/python

"""
ensemble.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB

import util

X_train, Y_train, X_test = util.loadData()

# SVM.
clf1 = svm.SVC(probability=True, C=2.915)

# AdaBoost.
clf2 = AdaBoostClassifier(n_estimators=170, learning_rate=0.85)

# Random forest with min leaf node size.
clf3 = RandomForestClassifier(n_estimators=100, criterion='gini',
                              min_samples_leaf=2)

# Random forest with max tree depth.
clf4 = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=62)

# Gradient boosting.
clf5 = GradientBoostingClassifier(n_estimators=115, max_depth=7)

# Logistic regression with L2 penalty.
clf6 = linear_model.LogisticRegression(penalty='l2', C=0.06, dual=False)

# Nearest neighbors.
clf7 = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance',
                                      leaf_size=3)

# Create ensemble using weighted average probabilities (soft voting) and
# predict on the test dataset.
eclf = VotingClassifier(estimators=[('svm', clf1), ('adaboost', clf2),
                        ('rf1', clf3), ('rf2', clf4), ('gb', clf5),
                        ('lr', clf6), ('knn', clf7)], voting='soft',
                        weights=[1,2,2,2,1,1,1])
eclf.fit(X_train,Y_train)
util.writeData(eclf.predict(X_test))
