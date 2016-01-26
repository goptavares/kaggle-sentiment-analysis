#!/usr/bin/python

"""
random_forests.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import util

# Load the training data.
X_train, Y_train, X_test = util.loadData()
clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             min_samples_split=1,random_state=0)
scores = cross_val_score(clf,X_train,Y_train) 
print("Cross validation mean: " + str(scores.mean()))

clf.fit(X_train, Y_train)
util.writeData(clf.predict(X_test))


"""
It seems that the cross validated score reaches a maximum score of ~0.7
"""