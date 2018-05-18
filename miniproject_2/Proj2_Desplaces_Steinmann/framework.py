from torch import FloatTensor, LongTensor, Tensor
import math

# Module Superclass

class Module(object):
    def forward(self, *input_):
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []

# ReLU Module

# This module represents the ReLU activation function
class ReLU(Module):
    def __init__(self):
        self.z = None
    
    # input_: the tensor outputed by the current layer
    def forward(self, input_):
        self.z = input_.clone()
        input_[input_ < 0] = 0
        return input_
        
    def backward(self, gradwrtoutput):
        da = gradwrtoutput
        tensor = self.z.clone()
        # g'(z)
        tensor[tensor > 0] = 1
        tensor[tensor < 0] = 0
        # dz[l]
        return da.mul(tensor)
        
    def param(self):
        return []
    
    def zero_grad(self):
        pass


# Tanh Module

class Tanh(Module):   
    def __init__(self):
        self.z = None
    
    # input_: the tensor outputed by the current layer
    def forward(self, input_):
        self.z = input_
        return input_.tanh()
        
    def backward(self, gradwrtoutput):
        da = gradwrtoutput
        # g'(z)
        g_prime = (1 - self.z.tanh().pow(2))
        # dz[l]
        return da.mul(g_prime)
        
    def param(self):
        return []
    
    def zero_grad(self):
        pass

# Linear Module

class Linear(Module):   
    def __init__(self, in_dim, out_dim):
        # keep track of the weigths, the biases and the output of the previous layer's activation function
        self.w = Tensor(out_dim,in_dim).normal_(0)
        self.b = Tensor(out_dim,1).normal_(0)
        self.x_previous_layer = None
        # init the gradient of the loss wrt w / b
        self.grad_w_sum = Tensor(self.w.size()).zero_()
        self.grad_b_sum = Tensor(self.b.size()).zero_()
    
    # input_: the output of the previous layer's activation function
    def forward(self, input_):
        self.x_previous_layer = input_
        return (self.w.mm(input_.t()) + self.b).t()
        
    def backward(self, gradwrtoutput):
        dz = gradwrtoutput.t()
        dw = dz.mm(self.x_previous_layer)
        db = dz
        # sum the gradients for the weights and biases
        self.grad_w_sum += dw
        self.grad_b_sum += db.sum(1).unsqueeze(1)
        return (self.w.t().mm(dz)).t()
        
    # returns a list of pairs, each composed of a parameter tensor and a gradient tensor
    # parameters: weights and biases
    def param(self):
        return [ (self.w, self.grad_w_sum), (self.b, self.grad_b_sum) ]
    
    def zero_grad(self):
        self.grad_w_sum.zero_()
        self.grad_b_sum.zero_()


# Sequential Module

# This module allows to combine several modules (layers, activation functions) in a basic sequential structure
class Sequential(Module):    
    def __init__(self, *layers_):
        self.modules = layers_
        
    # input_: the input data is a minibatch whose columns are features and lines are samples
    def forward(self, input_):
        x = input_
        for module in self.modules:
            x = module.forward(x)
        return x
        
    def backward(self, gradwrtoutput):
        x = gradwrtoutput
        for module in reversed(self.modules):
            x = module.backward(x)
        return x
        
    # returns a flatened list of each module's parameters
    # each parameter in the list is represented as a tuple containing the parameter tensor (e.g. w)
    # and the gradient tensor (e.g. dl/dw)
    def param(self):
        return [ p for module in self.modules for p in module.param() ]
    
    # sets the gradient of each layer to zero before the next batch can go through the network
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()


# MSE Loss Function

class LossMSE(Module): 
    def __init__(self):
        self.error = None
        
    def forward(self, preds, labels):
        self.error = preds - labels
        return self.error.pow(2).sum()
        
    def backward(self):
        return 2 * self.error
        
    def param(self):
        return []


# SGD Optimization

class optim_SGD(Module):
    # parameters: the parameters of the Sequential module
    def __init__(self, parameters, learning_rate):
        self.param = parameters #[ p.shallow() for tup in parameters for p in tup ]
        self.lr = learning_rate
        
    # performs a gradient step (SGD) for all parameters
    def step(self):
        for (p, grad_p) in self.param:
            p.sub_(self.lr*grad_p)


