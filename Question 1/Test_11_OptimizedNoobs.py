#Import the MLP class, the datasets and the libaries of Question 1
from run_11_OptimizedNoobs import *

#Import the best hyperparameters and parameters previously found
with open('bestHyperParams.pkl', 'rb') as f:
        bestHyperparams = pickle.load(f)
N_Opt, rho_Opt, sig_Opt = bestHyperparams[0][1]

with open('bestParams.pkl', 'rb') as f:
        bestP = pickle.load(f)
n = X_train.shape[1]
w_Opt, b_Opt, v_Opt = bestP[:n*N_Opt].reshape((n,N_Opt)), bestP[n*N_Opt:N_Opt+N_Opt*n],  bestP[N_Opt+N_Opt*n:]

def ICanGeneralize(X_new):
    mlpNet = MLP(X_train, y_train, X_test, y_test, N=N_Opt, sig=sig_Opt, rho=rho_Opt)
    mlpNet.w, mlpNet.b, mlpNet.v = w_Opt, b_Opt, v_Opt
    return mlpNet.pred(X_new)