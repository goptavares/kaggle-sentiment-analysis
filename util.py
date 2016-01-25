#!/usr/bin/python

"""
decision_trees.py
Authors: Gabriela Tavares,      gtavares@caltech.edu
         Juri Minxha,           jminxha@caltech.edu
"""
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os


def loadData():
    data = np.loadtxt('./training_data.txt', delimiter='|', skiprows=1)
    X_train = data[1:, 0:999]
    Y_train = data[1:, 1000]

    data = np.loadtxt('./testing_data.txt', delimiter='|', skiprows=1)
    X_test = data[1:, 0:999]

    return X_train, Y_train, X_test


def writeData(data):
    if not os.path.isdir(os.getcwd() + '/Solutions/'):
        os.mkdir(os.getcwd() + '/Solutions/')
    fileName = os.getcwd() + '/Solutions/' + str(datetime.datetime.now())
    np.savetxt(fileName, data, delimiter=',', header='ID,Prediction',
               fmt='%d', comments='')
