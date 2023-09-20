import numpy as np
from numba import jit, njit
import sys

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
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size




## gradient
def numerical_gradient(f, x, h=1e-4):
    # h = 1 * np.finfo(np.float32).eps
    # h = 1e-4
    grads = np.zeros_like(x)
    # print('x.size', x.size, 'x.shape', x.shape)
    iter = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not iter.finished:
        idx = iter.multi_index
        
        original_x = x[idx]
        
        x[idx] = float(original_x) + h
        # print(x[idx])
        # print(x[idx] == original_x)
        fxh1 = f(x)
        # print(fxh1)
        
        x[idx] = float(original_x) - h
        # print(x[idx])
        # print(x[idx] == original_x)
        fxh2 = f(x)
        # print(fxh2)
        
        grads[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = original_x
        iter.iternext()
        
    return grads


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)])
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)
    
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    return img[:, :, pad:H + pad, pad:W + pad]



## utils
def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용
    
    참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]