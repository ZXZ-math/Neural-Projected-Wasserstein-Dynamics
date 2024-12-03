import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np
 

# build a two layer relu network of the form:
# f(x,b) = \sum_{i=1}^m  w_i \sigma((x-  b_i)) + \sum_{i=m+1}^2m  w_i \sigma(-(x- b_i)) . 
# here x is in 1D 
# w_i's and b_i's are trainable parameters. There are 2*m of w_i and 2*m of b_i 

class two_layer(nn.Module):
    def __init__(self,m,bias_init,scale=True):
        super().__init__()
        self.m = m 
        self.bias_init = bias_init
        if scale:
            self.w_mul =  self.m
        else:
            self.w_mul = 1
        self.bias = nn.Parameter(torch.Tensor(2*m))
        self.w = nn.Parameter(torch.Tensor(2*m))
        self.reset_parameters()

    def reset_parameters(self):
        bias_initialbd = self.bias_init
        bias = torch.linspace(-bias_initialbd,bias_initialbd,self.m)
        self.bias.data[0:self.m] = bias
        self.bias.data[self.m:] = bias - 5e-6
        
        self.w.data[0:self.m] = 1/self.m*self.w_mul 
        self.w.data[self.m:] = -1/self.m*self.w_mul 
        
        
        
    def forward(self,x):
        # returns a 1 by n matrix where n is the number of samples 
        pos = torch.sum(self.w[:self.m]/self.w_mul * FF.relu((x - self.bias[:self.m])),1)
        neg = torch.sum(self.w[self.m:]/self.w_mul * FF.relu(-(x - self.bias[self.m:])),1)
        return pos+neg #+self.bias_init
    
    def df_db(self,x):
        # gradient of the neural network wrt b_i 
        # returns a n by 2m matrix where n is the number of samples 
        act_pos = x - self.bias[0:self.m]
        act_neg = self.bias[self.m:] - x 
        pos = -self.w[0:self.m]/self.w_mul * (act_pos >=0)
        neg = self.w[self.m:]/self.w_mul * (act_neg >=0)
        return torch.cat((pos,neg),1)
    
    def df_dw(self,x):
        # gradient of the neural network wrt w_i 
        # returns a n by 2m matrix where n is the number of samples 
        pos = FF.relu(x-self.bias[0:self.m])/self.w_mul
        neg = FF.relu(self.bias[self.m:]-x)/self.w_mul
        return torch.cat((pos,neg),1)
    
    def det_jac(self,x):
        # returns a 1 by n matrix where n is the number of samples 
        # each element is the det of the jacobian of f(input)
        act_pos = x - self.bias[0:self.m]
        act_neg = self.bias[self.m:] - x 
        pos = self.w[0:self.m]/self.w_mul * (act_pos >=0)
        neg = -self.w[self.m:]/self.w_mul * (act_neg >=0)
        return torch.sum(pos + neg,1)