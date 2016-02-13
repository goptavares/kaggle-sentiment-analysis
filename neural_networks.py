#!/usr/bin/python

"""
neural_networks.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import numpy as np
import operator

from sklearn.cross_validation import cross_val_score
from sknn.mlp import Classifier, Layer

import util

X_train, Y_train, X_test = util.loadData()

numNodesRange = np.arange(300, 1000, 50)
scoreCrossVal = list()
for numNodes in numNodesRange:
    clf = Classifier(layers=[Layer(type='Rectifier', units=numNodes),
                             Layer(type='Linear')],
                     learning_rate=0.01)
    scores = cross_val_score(clf, X_train, Y_train)
    scoreCrossVal.append(scores.mean())

index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimNumNodes = numNodesRange[index]
print("Optimal number of nodes: " + str(optimNumNodes))
