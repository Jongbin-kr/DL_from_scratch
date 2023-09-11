import numpy as np
from numba import jit, njit
## My own functions
## 228 µs ± 22 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
def step_function(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x

## 163 µs ± 34.6 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
def sigmoid(x):     
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

## 90.3 µs ± 4.85 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
def relu(x): 
    x[x <= 0] = 0
    return x

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
        
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))



### loss functions ###
def sum_squares_error(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum((y - t)**2)

def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    if y.ndim == 1:  ## if it is NOT batch
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t])) / batch_size


def numerical_gradient(f, x):
    h = 1e-4
    grads = np.zeros_like(x)
    # print('x.size', x.size, 'x.shape', x.shape)
    iter = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not iter.finished:
        idx = iter.multi_index
        
        original_x = x[idx]
        
        x[idx] = float(original_x) + h
        fxh1 = f(x)
        
        x[idx] = float(original_x) - h
        fxh2 = f(x)
        
        grads[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = original_x
        iter.iternext()
        
    return grads