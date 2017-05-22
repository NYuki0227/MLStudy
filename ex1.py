# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:44:59 2017

@author: Yuki
"""

import numpy as np
import pandas as pd


def costFunction (X, y, theta):
    m = y.size
    cost = np.sum((X.T @ theta - y) ** 2) / (2 * m)
    return cost

def gradientDecent (X, y, theta, alpha):
    m = y.size
    theta = theta - alpha * np.inner((X.T @ theta - y), X ) /  m
    return theta

if __name__ == '__main__':
    dat = np.loadtxt('ex1data1.txt', delimiter = ',', dtype = float)
    
    m = dat[:,0].size
    
    X = np.vstack((dat[:,0], np.ones(m)))
    y = dat[:,1]
    
    theta = (np.zeros(2))
    alpha = 0.01
    
    iteNum = 30
    cost = np.zeros(iteNum)
    
    for i in range(iteNum):
        cost[i] = costFunction(X, y, theta)
        theta = gradientDecent(X, y, theta, alpha)
    
    df = pd.DataFrame(cost)
    ax = df.plot()

    