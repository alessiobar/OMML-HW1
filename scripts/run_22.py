import time
import pickle
from functions_22 import *
import pandas as pd
Method = "Gradient Descent"

X_train, y_train, X_test, y_test = X[X.index.isin(idx)].reset_index(drop=True).T, y[y.index.isin(idx)].reset_index(drop=True).reshape((186, 1)),\
                                   X[~X.index.isin(idx)].reset_index(drop=True).T, y[~y.index.isin(idx)].reset_index(drop=True).reshape((64, 1)

if __name__ == '__main__':
    start = time.time()
    params, train_errors, y, num_grad, num_eval = fit(X_train, Y_train, N = 30, sigma = 1.2, r = 0.01, learning_rate = 0.01, number_of_iterations = 2000, supervised = False)
    with open ('weights22.pkl', 'wb') as f:
        pickle.dump(params,f)
    comp_time = time.time() - start
    tr_err = error_test(y, Y_train)
    y_tst, F, _  = forwardPropagation(X = X_test, params = params)
    tst_err = error_test(y_tst, Y_test)
    res = { "Number of neurons N chosen": params["N"],  "Value of spread chosen": params["sigma"], "Value of ro chosen": params["r_c"]/X_train.shape[1], "Optimization solver chosen": Method, "Number of function evaluations": num_eval, "Number of gradient evaluations": num_grad, "Time for optimizing the network": comp_time, "Training Error": tr_err, "Test Error": tst_err}
    print(res)
