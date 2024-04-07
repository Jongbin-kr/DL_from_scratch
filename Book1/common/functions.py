import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.shape[0])
        t = t.reshape(1, t.shape[0])
    
    batch_size = y.shape[0]
    
    return -np.sum(t * np.log(y)) / batch_size