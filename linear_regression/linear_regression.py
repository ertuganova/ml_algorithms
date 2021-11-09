import numpy as np

def solve_normal_equation(x, y):
    x_added = np.array([np.ones(len(x)), x.flatten()]).T
    weights = np.dot(np.dot(np.linalg.inv(np.dot(x_added.T, x_added)), x_added.T), y)   
    return weights, np.dot(x_added, weights)