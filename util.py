#!/usr/bin/python

"""
util.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os


def loadData(numFeatures=1000, normalize=False):
    data = np.loadtxt('./training_data.txt', delimiter='|', skiprows=1)
    X_train = data[:, 0:numFeatures]
    Y_train = data[:, 1000]

    data = np.loadtxt('./testing_data.txt', delimiter='|', skiprows=1)
    X_test = data[:, 0:numFeatures]

    if normalize:
        # Normalize X_train and X_test.
        for i in xrange(numFeatures):
            if np.std(X_train[:,i]) != 0:
                X_train[:,i] = ((X_train[:,i] - np.mean(X_train[:,i])) /
                                np.std(X_train[:,i]))
            if np.std(X_test[:,i]) != 0:
                X_test[:,i] = ((X_test[:,i] - np.mean(X_test[:,i])) /
                               np.std(X_test[:,i]))

    return X_train, Y_train, X_test


def writeData(predictions):
    numPoints = np.shape(predictions)[0]
    indices = np.arange(1, numPoints + 1).reshape(numPoints, 1)
    data = np.concatenate((indices, predictions.reshape(numPoints, 1)), axis=1)

    if not os.path.isdir(os.getcwd() + '/solutions/'):
        os.mkdir(os.getcwd() + '/solutions/')
    fileName = os.getcwd() + '/solutions/' + str(datetime.datetime.now())

    np.savetxt(fileName, data, delimiter=',', header='ID,Prediction',
               fmt='%d', comments='')
