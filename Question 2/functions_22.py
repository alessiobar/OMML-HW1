import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
def setParameters( X, N, sigma, r_v, r_c):

    input_size = 2 # number of neurons in input layer
    output_size = 2 # number of neurons in output layer.
    seed = 2027647
    np.random.seed(seed)
    V = np.random.randn(N, 1)
    C_ind = random.sample(range(0, X.shape[1]), N)
    C = X[:, C_ind]
    return {'C': C, 'V': V, "N": N, "sigma": sigma,"r_v":r_v, "r_c": r_c}

def rbf(x, ci, sigma):
    Z = np.sum((x - ci)**2)
    return np.exp(-Z/(sigma**2))

def forwardPropagation(X, params, num_eval = 0):
    num_eval += 1
    F = np.zeros((X.shape[1], params['N'] ))
    for i in range(X.shape[1]):
        for j in range(params['N']):
            F[i,j] = rbf(X[:,i].reshape(-1, 1),params['C'][:, j].reshape(-1, 1), params['sigma'])
    y = F @ params['V']
    return y ,  F, num_eval

def backPropagation(X, Y, params, F, y, supervised = True, num_grad = 0):
    num_grad += 2
    dV = F.transpose()@ F @ params['V'] -  F.transpose() @ Y + params['r_v']*params['V']
    dC = np.zeros((2,params['N']))
    if supervised == True:
        for k in range(params['N']):
            dPhi = np.zeros((2, X.shape[1]))
            for i in range(2):
                for j in range(X.shape[1]):
                    dPhi[i,j] = (X[i,j] - params['C'][i, k])*F[j,k]
            dPhi = dPhi*(2/(params['sigma']**2)) 
            dCk =  params['V'][k] *dPhi@(y-Y) + params['r_c']*np.reshape(params['C'][:, k],(2,1))
            dC[:,k] = dCk[:,0]
    return {'dC': dC, 'dV': dV}, num_grad

def updateParameters(gradients, params, learning_rate, supervised = True):
    C = params['C']
    if supervised == True:
        C = params['C'] - learning_rate*gradients['dC']
    V = params['V'] - learning_rate * gradients['dV']
    return {'C': C, 'V': V, 'N': params['N'], 'sigma': params['sigma'], "r_v":params['r_v'], "r_c":params['r_c']}

def error_test(y, Y):
    return np.sum((y - Y)**2)/(2*Y.shape[0])

def error(y, Y, params):
    return np.sum((y - Y)**2)/(2*Y.shape[0]) + params['r_v']*(np.sum(np.square(params['r_v'])))/2+params['r_c']*(np.sum(np.square(params['C'])))/2

def fit(X, Y, N = 20, sigma = 1.4, r = 0.01, learning_rate = 0.01, number_of_iterations = 2000, supervised = True, num_eval = 0, num_grad = 0):
    r_v = r
    r_c = r
    params = setParameters(X, N, sigma, r_v, r_c)
    tr_errors = []
    for j in range(number_of_iterations):
        y, F, num_eval = forwardPropagation(X, params, num_eval)
        tr_error = error(y, Y, params)
        gradients, num_grad = backPropagation(X, Y, params, F, y, supervised, num_grad)
        params = updateParameters(gradients, params, learning_rate, supervised)
        tr_errors.append(tr_error)
    return params, tr_errors, y, num_grad, num_eval
