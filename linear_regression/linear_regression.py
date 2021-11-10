import numpy as np

def solve_normal_equation(X, y):
    """
    Solve linear regression with normal equation
    :param X: matrix of objects (features)
    :param y: target vector
    :return: weights and y_hat for X
    """
    x_added = np.array([np.ones(len(X)), X.flatten()]).T
    weights = np.dot(np.dot(np.linalg.inv(np.dot(x_added.T, x_added)), x_added.T), y)   

    return weights, np.dot(x_added, weights)

def _compute_cost(X, y, w):
    """
    Compute cost function(least squared error)
    :param X: matrix of objects (features)
    :param y: target vector
    :param w: weights matrix
    :return: value of cost function
    """
    m = y.shape[0]
    J = 1/2*m*np.sum(np.dot(X, w) - y)**2
    
    return J

def gradient_descent(X, y, alpha, iterations, early_stopping=False):   
    """
    Solve linear regression with gradient descent (batch)
    :param X: matrix of objects (features)
    :param y: target vector
    :param alpha: step size/learning rate
    :param iterations: number of GD iterations
    :param early_stopping: stop GD if cost function value is not changing
    (or changing slowly)
    :return: weights, historical values of cost function, y_hat for X
    """
    m = y.shape[0]
    x_added = np.concatenate([np.ones((m, 1)), X], axis=1)
    w = np.zeros(x_added.shape[1])
    
    J_history = []
    
    for i in range(iterations):
        w = w - alpha*(1/m)*np.dot((np.dot(x_added, w) - y), x_added)
        J_history.append(_compute_cost(x_added, y, w))
        
        if early_stopping:
            if len(J_history)>2:
                if abs(J_history[-2] - J_history[-1])<0.01:
                    break

    return w, J_history, np.dot(x_added, w)