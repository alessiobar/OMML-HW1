""" QUESTION 1 """

#Import libraries
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

np.random.seed(2027647)
np.seterr(all='ignore')

idx = np.random.choice(X.shape[0],int(0.75*X.shape[0]),replace=False)
X_train, y_train, X_test, y_test = X[X.index.isin(idx)].to_numpy(), y[y.index.isin(idx)].to_numpy(),\
                                   X[~X.index.isin(idx)].to_numpy(), y[~y.index.isin(idx)].to_numpy()

#Build a MLP class to handle everything
class MLP:
    def __init__(self, X_train, y_train, X_test, y_test, N, sig, rho):
        self.x_train, self.y_train, self.x_test, self.y_test = X_train, y_train, X_test, y_test
        self.sig, self.rho, self.N = sig, rho, N
        self.n = self.x_train.shape[1]
        self.w, self.b, self.v = np.random.randn(self.n, N), np.repeat(0.01, N), np.random.randn(N) #set random inits

    def g(self, t, sig): #hyperbolic tangent function
        return (np.exp(2 * sig * t) - 1)/(np.exp(2 * sig * t) + 1)

    def pred(self, x): #forward pass computation
        t = x.dot(self.w) + self.b
        return self.g(t, self.sig).dot(self.v)

    def regLoss(self, x0): #regularized-loss function, ie. Eq. (1)
        N, n = self.N, self.n
        self.w, self.b, self.v = x0[:n * N].reshape((n, N)), x0[n * N:N + N * n], x0[N + N * n:]
        reg_loss = lambda x, y: np.sum(1/(2*x.shape[0])*(self.pred(x)-y)**2) + self.rho/2*(np.sum(self.w**2)+np.sum(self.b**2)+np.sum(self.v**2))
        return reg_loss(self.x_train, self.y_train)

    def dRegLoss(self, x0): #Gradient Evaluation for the regularized-loss, used as jac input for scipy.optimize.minimize
        n, N = self.n, self.N
        w, b, v = x0[:n * N].reshape((n, N)), x0[n * N:N + N * n], x0[N + N * n:]
        P = self.x_train.shape[0]
        x_train = self.x_train.reshape((P, 1, n))
        y_train = self.y_train.reshape((P, 1, 1))
        t = x_train @ w + b
        y_pred = self.g(t, self.sig) @ v
        y_pred = y_pred.reshape((P, 1, 1))
        grad_J = 1
        grad_y_pred = 1 / P * (y_pred - y_train) * grad_J
        grad_v = grad_y_pred @ self.g(t, self.sig)
        grad_g = grad_y_pred @ v.reshape((1, N))
        grad_t = grad_g * (4 * self.sig * np.exp(2 * self.sig * t) / (np.exp(2 * self.sig * t) + 1) ** 2)
        grad_b, grad_mul = grad_t, grad_t
        grad_w = np.transpose(x_train, axes=(0,2,1)) @ grad_mul
        grad_w, grad_b, grad_v = np.sum(grad_w, axis=0)+self.rho * w, np.sum(grad_b, axis=0)[0]+self.rho * b, np.sum(grad_v, axis=0)[0]+self.rho * v
        return np.concatenate((grad_w.flatten(), grad_b.flatten(), grad_v.flatten()))

    def optim(self): #optimization function invoking scipy.optimize.minimize
        N, n = self.N, self.n
        x0 = np.concatenate((self.w.flatten(), self.b.flatten(), self.v.flatten()))
        res = minimize(self.regLoss, x0, method='BFGS', jac=self.dRegLoss, options= {"maxiter":None, "disp":False})
        self.w, self.b, self.v = res.x[:n*N].reshape((n, N)), res.x[n*N:N+N*n], res.x[N+N*n:]
        return res

    def error(self, train=True): #training and test error evaluation function
        loss = lambda x, y: np.sum(1/(2*x.shape[0])*(self.pred(x)-y)**2)
        if train:
            l_train = loss(self.x_train, self.y_train)
            return l_train
        else:
            return loss(self.x_test, self.y_test)

if __name__ == '__main__':
    # GridSearch for the hyperparameters N, sig, rho:
    gridSearchActive = True
    # Grid defined by-hand for the three hyperparameters
    NList = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]
    sigList = [0.01, 0.15, 0.5, 1, 1.2, 1.4]
    rhoList = [1e-5, 1e-4, 1e-3]
    param_grid = [(n, r, s) for n in NList for r in rhoList for s in sigList]
    # Further split of train set into train and validation set (still 75-25%)
    idx = np.random.choice(X_train.shape[0], int(0.75 * X_train.shape[0]), replace=False)
    mask = np.ones((X_train.shape[0]), dtype=bool)
    mask[idx] = False
    X_tr, y_tr, X_val, y_val = X_train[idx], y_train[idx], X_train[mask], y_train[mask]

    if gridSearchActive:
        # Performance Evaluation for each hyperparameter triplet
        res = []
        for i in range(len(param_grid)):
            mlpNet = MLP(X_tr, y_tr, X_val, y_val, N=param_grid[i][0], rho=param_grid[i][1], sig=param_grid[i][2])
            startTime = datetime.now()
            opt = mlpNet.optim()
            execTime = datetime.now() - startTime
            lossOnVal, stats = opt.fun, [execTime, opt.message, opt.nfev, opt.nit, opt.njev, opt.status, opt.success]
            res.append([lossOnVal, param_grid[i], stats])
        bestHyperparams = [x for x in res if x[0] == min(x[0] for x in res)]
        N_Opt, rho_Opt, sig_Opt = bestHyperparams[0][1]
        # save hyperparams into a pickle file
        with open('bestHyperParams.pkl', 'wb') as f:
            pickle.dump(bestHyperparams, f)

    else:
        with open('bestHyperParams.pkl', 'rb') as f:
            bestHyperparams = pickle.load(f)
            N_Opt, rho_Opt, sig_Opt = bestHyperparams[0][1]

    # Build an MLP with the hyperparameters found
    mlpNet = MLP(X_train, y_train, X_test, y_test, N=N_Opt, sig=sig_Opt, rho=rho_Opt)
    t0 = datetime.now()
    finalRes = mlpNet.optim()
    execTime = (datetime.now() - t0).total_seconds()
    trErr = mlpNet.error()
    teErr = mlpNet.error(train=False)
    with open('bestParams.pkl', 'wb') as f:
        pickle.dump(finalRes.x, f)

    #Print statistics requested..
    print("The number of neurons N chosen:", N_Opt)
    print("The value of sigma chosen:", sig_Opt)
    print("The value of rho chosen:", rho_Opt)
    print("Optimization solver chosen:", "BFGS")
    print("Number of function evaluations:", bestHyperparams[0][2][2])
    print("Number of gradient evaluations:", bestHyperparams[0][2][4])
    print("Time for optimizing the network: {} seconds".format(round(execTime, 4)))
    print("Training Error:", trErr)
    print("Test Error:", teErr)

    #Plot of the Approximating function found in the specified region (functions taken from the Lab)
    def plotting(myFun):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x, y = np.linspace(-2, 2, 50), np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        Z = myFun(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
        ax.set_title("MLP Plot")
        plt.show()

    def apprFuncPlot(x, y):
        x, y = x.flatten(), y.flatten()
        Z = np.zeros(50*50)
        for i in range(50*50):
            Z[i] = mlpNet.pred(np.array([x[i], y[i]]))
        Z = Z.reshape((50,50))
        return Z

    plotting(apprFuncPlot)

    #Underfitting and Overfitting occurrence analysis by varying the hyperparameters

    def plotErr2(paramGrid, opt, param):
        N_Opt, rho_Opt, sig_Opt = opt
        plotTrErr, plotTeErr, axis = [], [], []

        for el in paramGrid[param]:
            print((el, rho_Opt, sig_Opt))
            if param =="N": mlpNet = MLP(X_tr, y_tr, X_val, y_val, N=el, rho=rho_Opt, sig=sig_Opt)
            elif param =="sig": mlpNet = MLP(X_tr, y_tr, X_val, y_val, N=N_Opt, rho=rho_Opt, sig=el)
            else: mlpNet = MLP(X_tr, y_tr, X_val, y_val, N=N_Opt, rho=el, sig=sig_Opt)
            mlpNet.optim()
            tr_error, test_error = mlpNet.error(), mlpNet.error(train=False)
            plotTrErr.append(tr_error)
            plotTeErr.append(test_error)
            axis.append(round(el, 3))
        plt.plot(axis, plotTrErr, 'o-', color="green", label="train error")
        plt.plot(axis, plotTeErr, 'o-', color="red", label="test error")
        plt.xlabel(plotAxisTitle[param])
        plt.ylabel('Error')
        plt.legend(loc="upper right")
        plt.title(plotTitle[param])
        plt.xticks(paramGrid[param])
        plt.show()

    paramGrid = {"N": NList, "sig": sigList, "rho": rhoList}
    plotTitle = {"N":"Test error and Train error vs number of neurons N", "sig":"Test error and Train error vs spread",
                 "rho":"Test error and Train error vs Regularization"}
    plotAxisTitle = {"N":"Number of neurons N", "sig":"Spread","rho":"Regularization p"}
    opt = (N_Opt, rho_Opt, sig_Opt)

    plotErr2(paramGrid, opt, "N")
    plotErr2(paramGrid, opt, "sig")
    plotErr2(paramGrid, opt, "rho")
