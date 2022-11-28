#Import the MLP class, the datasets and the libaries of Question 1
from run_bonus_OptimizedNoobs import *

# Function to pre-fetch the Network Parameters and the HyperParameters from a .pkl file
def ICanGeneralize(X_new):

    # Retrieving the hyperparameters used in the Question 1
    file = open('Bonus_hyperparam.pkl', 'rb')
    N, rho, sig = pickle.load(file)
    
    # Pre-fetching the network parameters 
    file = open('Bonus_params.pkl', 'rb')
    w, b, v = pickle.load(file)
	
    # Creating a MLP to be created using Two Block Decomposition method
    New_TBD_obj = Two_block_decomposition(N, rho, sig)
    
    # Loading the network parameters
    New_TBD_obj.load_parameters(w, b, v)
    
    # Returning the predictions of the model for the given samples 
    return New_TBD_obj.pred(X_new)
    

'''
#An example of how to call the function
df = pd.read_csv("Dataset.csv", header=None)
X, y = df.iloc[:,[0,1]], df.iloc[:,2]
print(ICanGeneralize(X))
'''
