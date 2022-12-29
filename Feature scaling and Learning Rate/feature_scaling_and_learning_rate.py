import numpy as np, copy, math
import matplotlib.pyplot as plt

# load dataset 
def load_house_data():
    data = np.loadtxt('houses.txt', delimiter=',', skiprows=1)
    X = data[:, :4]
    y = data[:, 4]
    return (X, y)

# call load dataset function
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']


# EDA
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()


def gradient_descent_houses(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X : (array_like Shape (m,n)    matrix of examples 
      y : (array_like Shape (m,))    target value of each example
      w_in : (array_like Shape (n,)) Initial values of parameters of the model
      b_in : (scalar)                Initial value of parameter of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    m = len(X)
    hist = {}
    hist['cost'] = []
    hist['params'] = []
    hist['grads'] = []
    hist['iter'] = []
    w = copy.deepcopy(w_in)
    b = b_in
    save_interval = np.ceil(num_iters / 10000)
    print('Iteration Cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  ')
    print('---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|')
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if not i == 0:
            if i % save_interval == 0:
                hist['cost'].append(cost_function(X, y, w, b))
                hist['params'].append([w, b])
                hist['grads'].append([dj_dw, dj_db])
                hist['iter'].append(i)
            if i % math.ceil(num_iters / 10) == 0:
                cst = cost_function(X, y, w, b)
                print(f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}")

    return (
     w, b, hist)

def compute_gradient_matrix(X, y, w, b):
    """
    Computes the gradient for linear regression 
 
    Args:
      X : (array_like Shape (m,n)) variable such as house size 
      y : (array_like Shape (m,1)) actual value 
      w : (array_like Shape (n,1)) Values of parameters of the model      
      b : (scalar )                Values of parameter of the model      
    Returns
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w. 
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b. 
                                  
    """
    m, n = X.shape
    f_wb = X @ w + b
    e = f_wb - y
    dj_dw = 1 / m * (X.T @ e)
    dj_db = 1 / m * np.sum(e)
    return (
     dj_db, dj_dw)

def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X : (ndarray): Shape (m,n) matrix of examples with multiple features
      w : (ndarray): Shape (n)   parameters for prediction   
      b : (scalar):              parameter  for prediction   
    Returns
      cost: (scalar)             cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2

    cost = cost / (2 * m)
    return np.squeeze(cost)

def run_gradient_descent(X, y, iterations=1000, alpha=1e-06):
    m, n = X.shape
    initial_w = np.zeros(n)
    initial_b = 0
    w_out, b_out, hist_out = gradient_descent_houses(X, y, initial_w, initial_b, compute_cost, compute_gradient_matrix, alpha, iterations)
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.2f}")
    return (
     w_out, b_out, hist_out)

#set alpha to 9.9e-7
_, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)

#set alpha to 9e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 9e-7)

def plot_cost_i_w(X, y, hist):
    ws = np.array([p[0] for p in hist['params']])
    rng = max(abs(ws[:, 0].min()), abs(ws[:, 0].max()))
    wr = np.linspace(-rng + 0.27, rng + 0.27, 20)
    cst = [compute_cost(X, y, np.array([wr[i], -32, -67, -1.46]), 221) for i in range(len(wr))]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(hist['iter'], hist['cost'])
    ax[0].set_title('Cost vs Iteration')
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('Cost')
    ax[1].plot(wr, cst)
    ax[1].set_title('Cost vs w[0]')
    ax[1].set_xlabel('w[0]')
    ax[1].set_ylabel('Cost')
    ax[1].plot(ws[:, 0], hist['cost'])
    plt.show()
    
plot_cost_i_w(X_train, y_train, hist)

#set alpha to 1e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 1e-7)

plot_cost_i_w(X_train,y_train,hist)


# ....Feature Scaling....
# z-score normalization
def zscore_normalization_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean(X, axis = 0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis = 0)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma
    
    return (X_norm, mu, sigma)

mu     = np.mean(X_train,axis=0)   
sigma  = np.std(X_train,axis=0) 
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma      

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:,0], X_mean[:,3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()

