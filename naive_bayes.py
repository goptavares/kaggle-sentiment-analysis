#!/usr/bin/python

"""
naive_bayes.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
from scipy.stats import binom_test
from sklearn.naive_bayes import MultinomialNB

import util

# Load the training data.
X_train, Y_train, X_test = util.loadData()

# Use a multinomial to model the distribution of the features since they're 
# word counts. 
clf = MultinomialNB(alpha=1.0, class_prior=[0.5,0.5], fit_prior=True)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_train)
accuracy = 100 * sum(predictions==Y_train) / len(Y_train)
p_value  = binom_test(sum(predictions == Y_train), len(Y_train), p=0.5)
print("Accuracy: %s" % accuracy + "%")
