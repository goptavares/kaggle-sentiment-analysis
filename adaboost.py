#!/usr/bin/python

"""
adaboost.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import matplotlib.pyplot as plt
import numpy as np
import operator

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

import util


X_train, Y_train, X_test = util.loadData(normalize=True)

# AdaBoost.
# Perform cross validation to find the optimal number of estimators and learning
# rate, in order to avoid overfitting.
numEstimatorsRange = range(160, 240, 10)
learningRateRange = np.arange(0.6, 1.3, 0.1)
scoreCrossVal = list()
score_matrix = np.zeros((len(numEstimatorsRange), len(learningRateRange)))
models = list()
i = 0
for numEstimators in numEstimatorsRange:
    j = 0
    for learningRate in learningRateRange:
        models.append((numEstimators, learningRate))
        print("Running model: " + str((numEstimators, learningRate)) + "...")
        clf = AdaBoostClassifier(n_estimators=numEstimators,
                                 learning_rate=learningRate)
        scores = cross_val_score(clf, X_train, Y_train)
        scoreCrossVal.append(scores.mean())
        score_matrix[i,j] = scores.mean()
        j +=1
    i += 1

index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimParams = models[index]
print("Optimal number of estimators and learning rate: " + str(optimParams))

x_labels = [str(x) for x in learningRateRange]
y_labels = [str(x) for x in numEstimatorsRange]
fig, ax = plt.subplots()
heatmap = ax.pcolor(score_matrix, cmap=plt.cm.rainbow, vmin=score_matrix.min(),
                    vmax=score_matrix.max())
ax.set_xticks(np.arange(score_matrix.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(score_matrix.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(x_labels, minor=False)
ax.set_yticklabels(y_labels, minor=False)
plt.xlabel('Learning rate')
plt.ylabel('Number of estimators')
plt.title('AdaBoost cross-validation scores')
plt.colorbar(mappable=heatmap, ax=ax)
plt.show()
