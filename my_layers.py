from my_functions import *
from dataclasses import dataclass

class ReLU():
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        out = relu(x)
        self.out = out
        # print('relu', x.shape, out.shape)
        return out
    
    def backward(self, dout):
        dout[self.out <= 0] = 0
        dx = dout
        return dx
    
    
    
class Sigmoid():
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out
    
    def backward(self, dout):
        dx = self.out * (1 - self.out) * dout
        return dx



class Affine():
    def __init__(self, W, b):
        self.W = W
        self.b = b
    
    def forward(self, x):
        # self.original_x_shape = x.shape
        # x = x.reshape(x.shape[0], -1)
        self.x = x
        out = self.x @ self.W + self.b
        self.out = out
        # print(f'x shape: {x.shape}, w shape: {self.W.shape}')
        # out = np.dot(self.x, self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        # dx = dx.reshape(*self.original_x_shape)
        return dx
        
        
class SoftmaxWithLoss():
    def __init__(self):
        pass
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)   ## softmax
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx
    
        
class Convolution:
    def __init__(self, W, b, stride=1, pad=0) -> None:
        self.W = W  ## filter
        self.b = b
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T ## spread filter
        out = col @ col_W + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        # print('Conv', x.shape, out.shape)
        
        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        
        self.db = np.sum(dout, axis=0)
        self.dW = self.col.T @ dout
        self.dW = self.dW.transpose(1,0).reshape(FN, C, FH, FW)
        
        dcol = dout @ self.col_W.T
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        
        return dx
        
    
@dataclass
class Pooling:
    pool_h: int
    pool_w: int
    stride: int = 0
    pad: int = 0
    
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)
        
        self.x = x
        self.arg_max = np.argmax(col, axis=1)
        # print('Pooling', self.x.shape, out.shape)
        
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0,2,3,1)
        
        pool_size = self.pool_h * self.pool_w
        
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

        
        
        




class BatchNormalization():
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None) -> None:
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        
        self.running_mean = running_mean
        self.running_var = running_var
        
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        
    
    def forward(self, x, train_flag=True):
        # self.input_shape = x.shape
        # if x.ndim != 2:
        #     N, C, H, W = x.shape
        #     x = x.reshape(N, -1)
        
        if self.running_mean is None:
            self.running_mean = np.zeros(x.shape[0])
            self.running_var = np.zeros(x.shape[0])
        
        if train_flag:
            mu = np.average(x, axis=0)
            xc = x - mu
            
            var = np.var(xc, axis=0)
            std = np.sqrt(var + 1e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.runnning_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.runnning_mean = self.momentum * self.running_mean + (1-self.momentum) * var
        
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + 1e-7)
            
        out = self.gamma * xn + self.beta
        # out = out.reshape(*self.input_shape)
        return out
    
    def backward(self, dout):
        # if dout.ndim != 2:
        #     N, C, H, W = dout.shape
        #     dout = dout.reshape(N, -1)
        
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * self.xn, axis=0)
        dxn = dout * self.gamma
        
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        
        dxc = dxn / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        
        dmu = np.sum(dxc, axis=0)
        
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
            
        # dx = dx.reshape(*self.input_shape)
        return dx
    
    
class Dropout:
    def __init__(self, dropout_ratio=0.5) -> None:
        self.dropout_ratio = dropout_ratio
        self.mask = None
        
    def forward(self, x, train_flag=True):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
        
    
    def backward(self, dout):
        # print(np.sum(self.mask))
        return dout * self.mask