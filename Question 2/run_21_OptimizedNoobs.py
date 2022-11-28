""" QUESTION 2 - OptimizedNoobs Group """

#Import the MLP class, the datasets and the libaries of Question 1
from run_11_OptimizedNoobs import *

#Import the best hyperparameters previously found
with open('bestHyperParams.pkl', 'rb') as f:
        bestHyperparams = pickle.load(f)
N_Opt, rho_Opt, sig_Opt = bestHyperparams[0][1]

#Build new class, inheriting from the MLP one
class TwoBlocksMethod(MLP):
    def __init__(self,X_train, y_train, X_test, y_test, N, sig, rho):
        super().__init__(X_train, y_train, X_test, y_test, N, sig, rho)

    def sqError(self, x0): #squared loss function
        self.v = x0
        sqLoss = lambda x, y: np.sum(1/(2*x.shape[0])*(self.pred(x)-y)**2)
        l_train = sqLoss(self.x_train, self.y_train)
        return l_train

    def dSqError(self, x0): #Gradient Evaluation for the squared loss, used as jac input for scipy.optimize.minimize
        n = self.n
        w, b, v = self.w, self.b, x0
        P = self.x_train.shape[0]
        x_train = self.x_train.reshape((P, 1, n))
        y_train = self.y_train.reshape((P, 1, 1))
        t = x_train @ w + b
        y_pred = self.g(t, self.sig) @ v
        y_pred = y_pred.reshape((P,1,1))
        grad_J = 1
        grad_y_pred = 1 / P * (y_pred - y_train) * grad_J
        grad_v = grad_y_pred @ self.g(t, self.sig)
        return np.sum(grad_v, axis=0)[0]

    def extremeLearning(self):
        np.random.seed(312321)
        self.w, self.b = np.random.uniform(-0.5, 0.5, [self.n, self.N]), np.random.uniform(-0.5, 0.5, [self.N])
        x0 = np.random.uniform(-5, 5, [self.N]) # v init
        res = minimize(self.sqError, x0, jac=self.dSqError, method='BFGS', options={"maxiter": None, "disp": False})
        self.v = res.x
        return res

if __name__ == '__main__':

    twoBlocks = TwoBlocksMethod(X_train, y_train, X_test, y_test, N=N_Opt, sig=sig_Opt, rho=rho_Opt)
    t0 = datetime.now()
    opt = twoBlocks.extremeLearning()
    execTime = (datetime.now() - t0).total_seconds()
    trErr = twoBlocks.error()
    teErr = twoBlocks.error(train=False)
    with open('bestVParam.pkl', 'wb') as f:
        pickle.dump((twoBlocks.w, twoBlocks.b, twoBlocks.v), f)

    #Print some statistics..
    print("The number of neurons N chosen:", N_Opt)
    print("The value of sigma chosen:", sig_Opt)
    print("The value of rho chosen:", rho_Opt)
    print("Optimization solver chosen:", "BFGS")
    print("Number of function evaluations:", opt.nfev)
    print("Number of gradient evaluations:", opt.njev)
    print("Time for optimizing the network: {} seconds".format(round(execTime,4)))
    print("Training Error:", round(trErr,5))
    print("Test Error:", round(teErr,5))


    #Plot of the Approximating function found in the specified region (functions taken from the Lab)
    def plotting(myFun):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x, y = np.linspace(-2, 2, 50), np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        Z = myFun(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
        ax.set_title("MLP Extreme Learning Plot")
        plt.show()

    def apprFuncPlot(x, y):
        x, y = x.flatten(), y.flatten()
        Z = np.zeros(50 * 50)
        for i in range(50 * 50):
            Z[i] = twoBlocks.pred(np.array([x[i], y[i]]))
        Z = Z.reshape((50, 50))
        return Z

    plotting(apprFuncPlot)