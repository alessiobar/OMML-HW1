import pickle
from functions_22 import *
from run_22 import *
def ICanGeneralize(X_test):
    #X_train,_, Y_train, _ = get_data()
    #params, train_errors, y, num_grad, num_eval = fit(X_train, Y_train, N = 30, sigma = 1.2, r = 0.01, learning_rate = 0.01, number_of_iterations = 2000, supervised = False)
    with open('weights22.pkl', 'rb') as f:
        params = pickle.load(f)
    y_tst, F, _  = forwardPropagation(X = X_test, params = params)
    return y_tst
print(ICanGeneralize(X_train)- Y_train)
