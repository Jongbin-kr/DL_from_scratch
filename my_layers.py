from my_functions import *

class ReLU():
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        out = relu(x)
        self.out = out
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
        self.x = x
        # self.out = out
        # print(f'x shape: {x.shape}, w shape: {self.W.shape}')
        out = x @ self.W + self.b
        return out
    
    def backward(self, dout):
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
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
    
        

class BatchNormalization():
    def __init__(self) -> None:
        self.running_mean = None
        self.running_var = None
    
    def forward(self, x):
        x -= np.average(x)
        x /= np.sqrt(np.var(x) + 1e7)
        
        if self.running_mean is None:
            # self.running_mean = 
            pass
        return x
    
    def backward(self, dout):
        pass
    
    
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
        print(np.sum(self.mask))
        return dout * self.mask