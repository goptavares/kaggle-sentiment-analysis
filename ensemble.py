#!/usr/bin/python

"""
ensemble.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
							  GradientBoostingClassifier,
							  RandomForestClassifier, VotingClassifier)

import util


X_train, Y_train, X_test = util.loadData(normalize=True)

# SVM.
clf1 = svm.SVC(probability=True, C=2.551, kernel='rbf')

# AdaBoost.
clf2 = AdaBoostClassifier(n_estimators=220, learning_rate=0.9)

# Random forest with min leaf node size.
clf3 = RandomForestClassifier(n_estimators=400, criterion='gini',
                              min_samples_leaf=1)

# Random forest with max tree depth.
clf4 = RandomForestClassifier(n_estimators=400, criterion='gini', max_depth=53)

# Gradient boosting.
clf5 = GradientBoostingClassifier(n_estimators=150, max_depth=7)

# Logistic regression with L1 penalty.
clf6 = linear_model.LogisticRegression(penalty='l1', C=0.35, dual=False)

# Logistic regression with L2 penalty.
clf7 = linear_model.LogisticRegression(penalty='l2', C=0.03, dual=False)

# Nearest neighbors.
clf8 = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance',
                                      leaf_size=3)

# QDA.
clf9 = QuadraticDiscriminantAnalysis(reg_param=0.007)

clf10 = ExtraTreesClassifier(n_estimators=400, criterion='gini',
                             min_samples_leaf=3)

# Create ensemble using weighted average probabilities (soft voting) and
# predict on the test dataset.
eclf = VotingClassifier(estimators=[('svm', clf1), ('adaboost', clf2),
                        			('rf1', clf3), ('rf2', clf4), ('gb', clf5),
                        			('lr1', clf6), ('lr2', clf7), ('knn', clf8),
                        			('qda', clf9), ('erf', clf10)],
                        voting='soft', weights=[1, 1, 3, 3, 3, 1, 1, 1, 0.5, 2])
eclf.fit(X_train,Y_train)
scores = cross_val_score(eclf, X_train, Y_train)
print("Cross validation score: " + str(scores.mean()))
util.writeData(eclf.predict(X_test))
