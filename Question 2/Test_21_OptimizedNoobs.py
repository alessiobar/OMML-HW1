# Import the MLP class, the datasets and the libaries of Question 1
from run_21_OptimizedNoobs import *

#Import the best parameter v previously found
with open('bestVParam.pkl', 'rb') as f:
    w_notOpt, b_notOpt, v_Opt = pickle.load(f)

def ICanGeneralize(X_new):
    twoB = MLP(X_train, y_train, X_test, y_test, N=N_Opt, sig=sig_Opt, rho=rho_Opt)
    twoB.w, twoB.b, twoB.v = w_notOpt, b_notOpt, v_Opt
    return twoB.pred(X_new)