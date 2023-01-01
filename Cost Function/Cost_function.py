import numpy as np
import matplotlib.pyplot as plt

# Training data
x_train = np.array([1.0, 2.0])    #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])  #(price in 1000s of dollars)

# Initializing weights and bias 
w = 250
b = 4

# Computing Cost function

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]
    
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum
    
    return total_cost

cost_f = compute_cost(x_train, y_train, w, b)

print(cost_f)