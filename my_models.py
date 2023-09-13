from collections import OrderedDict

import numpy as np

from my_functions import *
from my_layers import *



class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01) -> None:
        
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # self.layers['Relu2'] = ReLU()
        
        self.last_layer = SoftmaxWithLoss()
    
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    
    def loss(self, x, t):
        y = self.predict(x)

        loss = self.last_layer.forward(y, t)
        return loss
    
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:     ## 원-핫 라벨인 경우, 다시 1열짜리 매트릭스로 전환.
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
    
    def gradient(self, x, t):
        
        self.loss(x, t)
        
        dout = 1
        dout = self.last_layer.backward(dout)
        reversed_layers = reversed(self.layers.values())
        for layer in reversed_layers:
            # print(layer)
            dout = layer.backward(dout)
            
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads
    
    
    
def batch_mask_loader(data: np.ndarray, batch_size=100) -> np.ndarray:
    batch_indexes= np.arange(len(data))
    np.random.shuffle(batch_indexes)
    while batch_indexes.any():
        batch_indexes = batch_indexes[batch_size:]
        
        batch_mask = batch_indexes[:batch_size]
        # print(batch)
        yield batch_mask
