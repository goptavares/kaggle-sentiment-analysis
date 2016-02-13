#!/usr/bin/python

"""
nearest_neighbors.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import matplotlib.pyplot as plt
import numpy as np
import operator

from sklearn import neighbors
from sklearn.cross_validation import cross_val_score

import util


X_train, Y_train, X_test = util.loadData(normalize=True)

# Perform cross validation to find the optimal number of neighbors.
numNeighborsRange = range(1, 12)
leafSizeRange = range(1, 12)
scoreCrossVal = list()
score_matrix = np.zeros((len(numNeighborsRange), len(leafSizeRange)))
models = list()
i = 0
for numNeighbors in numNeighborsRange:
    j = 0
    for leafSize in leafSizeRange:
        models.append((numNeighbors, leafSize))
        print("Running model: " + str((numNeighbors, leafSize)) + "...")
        clf = neighbors.KNeighborsClassifier(n_neighbors=numNeighbors,
                                             weights='distance',
                                             leaf_size=leafSize)
        scores = cross_val_score(clf, X_train, Y_train)
        scoreCrossVal.append(scores.mean())
        score_matrix[i,j] = scores.mean()
        j += 1
    i += 1

index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimParams = models[index]
print("Optimal number of neighbors and leaf size: " + str(optimParams))

x_labels = [str(x) for x in leafSizeRange]
y_labels = [str(x) for x in numNeighborsRange]
fig, ax = plt.subplots()
heatmap = ax.pcolor(score_matrix, cmap=plt.cm.rainbow, vmin=score_matrix.min(),
                    vmax=score_matrix.max())
ax.set_xticks(np.arange(score_matrix.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(score_matrix.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(x_labels, minor=False)
ax.set_yticklabels(y_labels, minor=False)
plt.xlabel('Leaf size')
plt.ylabel('Number of neighbors')
plt.title('Nearest neighbors cross-validation scores')
plt.colorbar(mappable=heatmap, ax=ax)
plt.show()
