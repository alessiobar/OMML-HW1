
#Importing the necessary libraries

import pandas as pd, numpy as np
from scipy.optimize import minimize
import scipy.optimize
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

X_train, y_train, X_test, y_test = X[X.index.isin(idx)].reset_index(drop=True), y[y.index.isin(idx)].reset_index(drop=True),\
                                   X[~X.index.isin(idx)].reset_index(drop=True), y[~y.index.isin(idx)].reset_index(drop=True)


#Build a two block decomposition method class to handle everything
class Two_block_decomposition:
    def __init__(self, N=None, rho=None, sig=None, K=None, seed=None):
      '''
      ######################################
                Hyper parameters 
      ######################################
      '''
      # N: The number of the neurons in the hidden layer 
      self.sig, self.rho, self.N = sig, rho, N 

      # The number of the iterations 
      self.K = K

      # The number of the functions evaluated 
      self.nfunc = 0 

      # The number of the gradients evaluated 
      self.ngrad = 0

      # The number of the outer iterations 
      # As due to the stopping early criteria the model may stop sooner 
      self.iterations = 0 

      # The seed that we want to use for the random number generation
      self.seed = seed

      # The error on the training set before starting the procedure 
      self.baseError = 0 

    def load_parameters(self, W, B, V): 

      # Preload the paramters 
      self.w, self.b, self.v = W, B, V

      # Set the number of the features (n)
      self.n = self.w.shape[0]

    #hyperbolic tangent function, the activation function
    def g(self, t, sig):
      return (np.exp(2 * sig * t) - 1)/(np.exp(2 * sig * t) + 1)
    
    # Loss when we have the hidden layer weights, bias fixed
    def loss_w2(self, x0): 
      self.v = x0
      Loss = lambda x, y: np.sum(1/(2*x.shape[0])*(self.pred(x)-y)**2) + self.rho/2 * np.sum(self.v**2)
      loss = Loss(self.x_train, self.y_train)
      return loss

    # Loss when we have the output layer weights fixed
    def loss_w1_bias(self, x0): 
      
      # N: the number of the neurons 
      # n: the number of the features 
      N, n = self.N, self.n

      # Get the parameters of the model 
      self.w, self.b = x0[:n * N].reshape((n, N)), x0[n * N:N + N * n]
      Loss = lambda x, y: np.sum(1/(2*x.shape[0])*(self.pred(x)-y)**2) + self.rho/2 * (np.sum(self.w**2)+np.sum(self.b**2))
      loss = Loss(self.x_train, self.y_train)
      return loss

    # Computes the gradient of the weights in the hidden layer and the bias with respect to the loss function
    def dRegLoss_w1_bias(self, x0): 
      
      # N: the number of the neurons in the hidden layer 
      # n: the number of the features in the training data 
      n, N = self.n, self.N

      # Take the current values of the parameters of the network 
      # The weights of the hidden layer, the bias added, the weights of the ouput layer 
      w, b, v = x0[:n * N].reshape((n, N)), x0[n * N:N + N * n], self.v

      # Take the number of the samples that we have in the training data
      P = self.x_train.shape[0]
      x_train = self.x_train.reshape((P, 1, n))
      y_train = self.y_train.reshape((P, 1, 1))

      # Get the output of the first hidden layer, the multiplication of the weights + the bias of the neuron 
      t = x_train @ w + b

      # Get the prediction of the model, the output of the activation function and the output weights v
      y_pred = self.g(t, self.sig) @ v
      y_pred = y_pred.reshape((P,1,1))

      # Computing the gradient of the parameters with repspect to the loss function
      grad_J = 1
      grad_y_pred = 1 / P * (y_pred - y_train) * grad_J
      grad_g = grad_y_pred @ v.reshape((1, N))
      grad_t = grad_g * (4 * self.sig * np.exp(2 * self.sig * t) / (np.exp(2 * self.sig * t) + 1) ** 2)
      grad_b, grad_mul = grad_t, grad_t
      grad_w = np.transpose(x_train, axes=(0,2,1)) @ grad_mul
      grad_w, grad_b= np.sum(grad_w, axis=0)+self.rho * w, np.sum(grad_b, axis=0)[0]+self.rho * b
      return np.concatenate((grad_w.flatten(), grad_b.flatten()))

    # Two block decomposition method 
    def TBD(self): 
      
       # The number of the functions evaluated 
      self.nfunc = 0 

      # The number of the gradients evaluated 
      self.ngrad = 0

      # N: the number of the neurons in the hidden layer 
      # n: the number of hte features in the training data 
      N, n = self.N, self.n

      # Keep the previous values of the parameters to have them in case the error on the validation set increases 
      prev_w, prev_b, prev_v, prev_val_acc = self.w, self.b, self.v, self.error(self.pred(self.x_val), self.y_val)

      # Iterate K times to update the parameters of the network
      for iter in range(self.K): 
        
        # 1 - Keeping the weights fixed and update the output weights 
        # Updating the values of the output weights 
        Bounds = [(-100, 100)] * self.v.shape[0]
        res = scipy.optimize.dual_annealing(func = self.loss_w2, bounds = Bounds, maxiter=500)
        self.v = res.x

        # How many functions have been evaluated 
        self.nfunc += res.nfev
        # How many gradients have been computed 
        self.ngrad += res.njev
        
        # 2- Keeping the output weights fixed and update the hidden layer weights and bias
        # Updating the values of hidden layer weights and the bias 
        x0 = np.concatenate((self.w.flatten(), self.b.flatten()))
        res = minimize(self.loss_w1_bias, x0, method='BFGS', jac=self.dRegLoss_w1_bias, options= {"maxiter":2000, "disp":False}, 
                      tol=1e-8)
        self.w, self.b = res.x[:n*N].reshape((n, N)), res.x[n*N:N+N*n]

        # How many functions have been evaluated 
        self.nfunc += res.nfev
        # How many gradients have been computed 
        self.ngrad += res.njev

        # Get the error of the model on the validation set 
        current_val_acc = self.error(self.pred(self.x_val), self.y_val)

        # Number of the performed iterations
        self.iterations+=1
        
        # If the error on the validation set got increased, terminate the process and use the previous parameters  
        if current_val_acc >= prev_val_acc: 
          # Put the previous weights as the weights of the model 
          self.w, self.b, self.v = prev_w, prev_b, prev_v
          return True
        
        else: 
          # Keep track of the previous values of the parameters
          prev_w, prev_b, prev_v= self.w, self.b, self.v
          
          # Keep track of the previous error of the model on the validation set 
          prev_val_acc = current_val_acc        
      
      # Return trained successfully
      return True 

    # Building the model using the training data 
    def fit(self, X_train, y_train): 

      '''
      ######################################
                Receiving the data
      ######################################
      '''
      # The training and the test set 
      self.x_train, self.y_train = X_train, y_train

      # Take 20% of the training data as validation set for the stopping criteria 
      
      self.Take_val(0.2)

      self.x_train, self.y_train , self.x_val, self.y_val = self.x_train.to_numpy(), self.y_train.to_numpy(), self.x_val.to_numpy(), self.y_val.to_numpy()
      '''
      ######################################
              Setting the parameters 
      ######################################
      '''

      # The number of the features in the trainig set 
      self.n = self.x_train.shape[1]

      # The randomized parameters of the network 
      # The weights of the hidden layer, the bias, the weights of the output layer 
      self.w, self.b, self.v = np.random.randn(self.n, self.N), np.repeat(0.01, self.N), np.random.randn(self.N) 

      # Take the error of the model before starting the optimization process 
      self.baseError = self.error(self.pred(X_train), y_train)

      # Updating the parameters of the model using the provided data 
      return self.TBD()

    # The forward pass to compute the predictions of the samples
    def pred(self, x): 
      t = x.dot(self.w) + self.b
      return self.g(t, self.sig).dot(self.v)

    def error(self, y_pred, y_true): 
      loss = lambda x, y: np.sum(1/(2*x.shape[0])*(x-y)**2)
      return loss(y_pred, y_true)

    def plotApproximationFunc(self, title = 'Two block decomposition', save_fig = True, path = 'TBD'):
      fig = plt.figure()
      ax = plt.axes(projection='3d')
      x, y = np.linspace(-2, 2, 50), np.linspace(-3, 3, 50)
      X, Y = np.meshgrid(x, y)
      Z = self.approximationFunction(X, Y)
      ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
      ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
      ax.set_title(f'{title} plot')
      if save_fig: 
        plt.savefig(f'{path}.png')
      plt.show()

    def approximationFunction(self, x, y):
      x, y = x.flatten(), y.flatten()
      Z = np.zeros(50*50)
      for i in range(50*50):
          Z[i] = self.pred(np.array([x[i], y[i]]))
      Z = Z.reshape((50,50))
      return Z

    # Storing the value of the parameters of the network
    def save_optimized_params(self, path = 'TBD_params'):
      with open(f'{path}.pkl', 'wb') as f:
        pickle.dump([self.w, self.b, self.v], f)

    def save_hyperparams(self, path = 'TBD_hyperparam'): 
      with open(f'{path}.pkl', 'wb') as f: 
        pickle.dump([self.N, self.rho, self.sig], f)
    
    # Giving some statistics about the last training 
    def last_train_stat_full(self): 
      print(f'The number of the neurons in the hidden layer: \'{self.N}\'')
      print(f'The value of sigma: \'{self.sig}\'')
      print(f'The value of rho: \'{self.rho}\'')
      print(f'The chosen optimization solver: \'BFGS\'')
      print(f'The number of the outer iterations (subproblems solved): \'{self.iterations}\'')
      print(f'The number of the evaluated functions: \'{self.nfunc}\'')
      print(f'The number of the evaluated gradients: \'{self.ngrad}\'')
      print(f'The error of the model before starting the optimization process: \'{self.baseError}\'')
      return [self.N, self.sig, self.rho, 'BFGS', self.iterations, self.nfunc, self.ngrad, self.baseError]

    def last_train_stat_optimize(self): 
      print(f'The number of the outer iterations (subproblems solved): \'{self.iterations}\'')
      print(f'The number of the evaluated functions: \'{self.nfunc}\'')
      print(f'The number of the evaluated gradients: \'{self.ngrad}\'')
      print(f'The error of the model before starting the optimization process: \'{self.baseError}\'')
      return [self.iterations, self.nfunc, self.ngrad, self.baseError]
  
    # Take a portion of the data as the validation set for the stopping criteri
    def Take_val(self, portion): 
      X, y = self.x_train, self.y_train
      np.random.seed(self.seed)
      idx = np.random.choice(X.shape[0],int((1 - portion) *X.shape[0]),replace=False)
      self.x_train, self.y_train, self.x_val, self.y_val = X[X.index.isin(idx)].reset_index(drop=True), y[y.index.isin(idx)].reset_index(drop=True),\
                                   X[~X.index.isin(idx)].reset_index(drop=True), y[~y.index.isin(idx)].reset_index(drop=True)
if __name__ == '__main__': 


	# The matricola of one of the groupmmates
	Seed = 2027647	
	X, y = df.iloc[:,[0,1]], df.iloc[:,2]

	# The hyper parameters of the best model that we got in the Question 1			                   
	N, rho, sig = [35, 1e-05, 1.4]

	# The number of the iterations in the two block decomposition approach
	K = 20

	# Create a Two block decomposition approach object
	TBD_obj = Two_block_decomposition(N=N, rho=rho, sig=sig, K=K, seed=Seed)

	# Begin the optimizing the network's parameters 
	opt = TBD_obj.fit(X, y)

	# Check the error on the training set 
	trErr = TBD_obj.error(TBD_obj.pred(X), y)
	print(f"Training Error: \'{round(trErr, 5)}\'")
	TBD_obj.save_optimized_params('Bonus_params')
	TBD_obj.save_hyperparams('Bonus_hyperparam')
