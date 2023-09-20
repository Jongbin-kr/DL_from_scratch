from collections import OrderedDict

import numpy as np

from my_functions import *
from my_layers import *



class MultiLayerNet():
    def __init__(self, input_size, hidden_size_list, output_size,
                 h=1e-4,
                 weight_init_std='relu', activation='relu', 
                 weight_decay_method='pass', weight_decay_lambda=0,
                 batch_norm=False,
                 dropout=False, dropout_ratio=0.1) -> None:
        
        self.h = h
        
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        
        self.weight_decay_method = weight_decay_method
        self.weight_decay_lambda = weight_decay_lambda
        
        self.params = {}
        self.__init_weight(weight_init_std)
        
        self.batch_norm = batch_norm
        self.dropout = dropout
        
        
        activation_layers = {'sigmoid': Sigmoid, 'relu': ReLU, }
        self.layers = OrderedDict()
        for idx in range(1, len(self.hidden_size_list)+1):
            self.layers[f'Affine{idx}'] = Affine(self.params[f'W{idx}'], self.params[f'b{idx}'])

            if self.batch_norm == 'before':
                self.params[f'gamma{idx}'] = np.ones(hidden_size_list[idx-1])
                self.params[f'beta{idx}'] = np.zeros(hidden_size_list[idx-1])
                self.layers[f'BatchNorm{idx}'] = BatchNormalization(self.params[f'gamma{idx}'], self.params[f'beta{idx}'])                        

                
                
            self.layers[f'Activation{idx}'] = activation_layers[activation]()
            # print(f'Affine{idx} and Activation{idx} created')
            if self.batch_norm == 'after':
                self.params[f'gamma{idx}'] = np.ones(hidden_size_list[idx-1])
                self.params[f'beta{idx}'] = np.zeros(hidden_size_list[idx-1])
                self.layers[f'BatchNorm{idx}'] = BatchNormalization(self.params[f'gamma{idx}'], self.params[f'beta{idx}'])            

        
            if self.dropout:
                self.layers[f'Dropout{idx}'] = Dropout(dropout_ratio=dropout_ratio)
        self.layers[f'Affine{idx+1}'] = Affine(self.params[f'W{idx+1}'], self.params[f'b{idx+1}'])
        self.last_layer = SoftmaxWithLoss()
    
    
    
    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'tanh', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            else:
                scale = weight_init_std
            self.params[f'W{idx}'] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params[f'b{idx}'] = np.zeros(all_size_list[idx])
            
    
    
    def predict(self, x, train_flag=False):
        for key, layer in self.layers.items():
            if ('Dropout' in key) or ('BatchNorm' in key):
                x = layer.forward(x, train_flag)
            # print(f'layer name: {key}')
            else:
                x = layer.forward(x)
        return x
    
    
    
    def loss(self, x, t, train_flag=False):
        y = self.predict(x, train_flag=train_flag)
        
        weight_decay = 0
        if self.weight_decay_method == 'L2':
            for idx in range(1, len(self.hidden_size_list)+2):
                W = self.params[f'W{idx}']
                weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        
        elif self.weight_decay_method == 'L1':
            for idx in range(1, len(self.hidden_size_list)+2):
                W = self.params[f'W{idx}']
                weight_decay += 0.5 * self.weight_decay_lambda * np.sum(np.abs(W))
                
                
        loss = self.last_layer.forward(y, t)
        return loss + weight_decay
    
    
    
    def accuracy(self, x, t):
        y = self.predict(x, train_flag=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:     ## 원-핫 라벨인 경우, 다시 1열짜리 매트릭스로 전환.
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    
    def numerical_gradient(self, x, t, h=None):
        
        if h is None:
            h = self.h
        
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'], h)
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'], h)
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'], h)
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'], h)
        
        return grads
    
    
    def gradient(self, x, t):
        
        self.loss(x, t, train_flag=True)
        
        dout = 1
        dout = self.last_layer.backward(dout)
        
        reversed_layers = reversed(self.layers.values())
        for layer in reversed_layers:
            # print(layer)
            dout = layer.backward(dout)
            
        grads = {}
        for idx in range(1, len(self.hidden_size_list)+2):
            if self.weight_decay_method in ('L2', 'pass'):
                grads[f'W{idx}'] = self.layers[f'Affine{idx}'].dW + (self.weight_decay_lambda * self.params[f'W{idx}'])
                
            elif self.weight_decay_method == 'L1':
                # abs_gradient = np.zeros_like(self.params[f'W{idx}'])
                # it = np.nditer(self.params[f'W{idx}'], flags=['multi_index'])
                # while not it.finished:
                #     multi_idx = it.multi_index
                #     abs_gradient[multi_idx] = (1 if self.params[f'W{idx}'][multi_idx] > 0 else -1)
                #     it.iternext()
                positive_mask, negative_mask = self.params[f'W{idx}'] > 0, self.params[f'W{idx}'] <= 0
                abs_gradient = np.ones_like(self.params[f'W{idx}'])
                abs_gradient[positive_mask] = self.weight_decay_lambda
                abs_gradient[negative_mask] = -self.weight_decay_lambda
                grads[f'W{idx}'] = self.layers[f'Affine{idx}'].dW + 0.5 * self.weight_decay_lambda * abs_gradient
            
            grads[f'b{idx}'] = self.layers[f'Affine{idx}'].db
            
            if self.batch_norm and idx != len(self.hidden_size_list)+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta            
        
        return grads
    
    



    
def batch_mask_loader(data: np.ndarray, batch_size=100, shuffle=True) -> np.ndarray:
    batch_indexes= np.arange(len(data))
    if shuffle:
        np.random.shuffle(batch_indexes)
    while batch_indexes.any():
        batch_mask = batch_indexes[:batch_size]
        batch_indexes = batch_indexes[batch_size:]
        # print(batch)
        yield batch_mask





## optimizers
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
            
            
class Momentum():
    def __init__(self, lr=0.01, momentum=0.9) -> None:
        self.lr = lr
        self.v = None
        self.momentum = momentum
        
    def update(self, params, grads):
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = (self.momentum * self.v[key]) - (self.lr * grads[key])
            params[key] += self.v[key]
            


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key]**2
            
            update_value = self.lr / (np.sqrt(self.h[key]) + 1e-7) * grads[key]
            params[key] -= update_value
            
            
            
class RMSProp:
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.h = None
        self.decay_rate = decay_rate
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1-self.decay_rate) * grads[key] * grads[key]
            
            update_value = self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            params[key] -= update_value
            
            
            
class Adam():
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.iter = 1
    
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():        
            self.m[key] = self.beta1 * self.m[key] + ((1.0 - self.beta1) * grads[key]) 
            # self.m[key] /= (1.0 - self.beta1**self.iter)
            self.v[key] = self.beta2 * self.v[key] + ((1.0 - self.beta2) * grads[key]**2) 
            # self.v[key] /= np.sqrt(1.0 - self.beta2**self.iter)
            
            params[key] -= self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter) * self.m[key] / (np.sqrt(self.v[key]) + 1e-7) 
        
        self.iter += 1
        
        
        
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
