import numpy as np

def numerical_gradient(f, x):
    grad = np.zeros_like(x)
    h = 1e-4
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    
    return grad

        
        