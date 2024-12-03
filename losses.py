import numpy as np
import torch
from torch import nn


class Relative_Entropy(nn.Module):

    def __init__(self, potential, dimension):
        super().__init__()

        self.potential = potential
        self.dimension = dimension

    def forward(self, zk, log_jacobians, D = 1):

        sum_of_log_jacobians = sum(log_jacobians)
    
        return (-sum_of_log_jacobians * D + self.potential(zk, self.dimension)).mean()
    

class Relative_Entropy_FC(nn.Module):
    def __init__(self, potential, dimension,D,weights):
        super().__init__()

        self.potential = potential
        self.dimension = dimension
        self.D = D
        self.weights = weights

    def forward(self, zk, det_jacobian, pz = 0):

        sum_of_log_jacobians = torch.log(1/torch.abs(det_jacobian))
        if self.weights[0]>0:
            return ((sum_of_log_jacobians * self.D + self.potential(zk, self.dimension))*self.weights.T).sum()
        if type(self.potential) == int or type(self.potential) == float:
            return (sum_of_log_jacobians * self.D).mean()
        else: 
            return (sum_of_log_jacobians * self.D + self.potential(zk, self.dimension)).mean()
    

    


class Linear_loss(nn.Module):
    def __init__(self, potential, dimension,weights):
        super().__init__()

        self.potential = potential
        self.dimension = dimension
        self.weights = weights


    def forward(self, zk):
        if self.weights[0]>0:
            return ((self.potential(zk, self.dimension))*self.weights.T).sum()
        return (self.potential(zk, self.dimension)).mean()

class Porous_medium_loss(nn.Module):
    def __init__(self,m=2,dimension=1):
        super().__init__()
        self.m = m
        self.dimension = dimension

    def forward(self,det_jacobian,pz):
        u_hat = 1/(self.m-1) * (pz/torch.abs(det_jacobian))**(self.m-1)
        return u_hat.mean()
    
class Interaction_loss(nn.Module):
    def __init__(self,W,n,D=1):
        super().__init__()
        self.W = W
        self.D = D
        self.n = n
    def forward(self,v,det_jacobian=1):
        sum_of_log_jacobians = torch.log(1/torch.abs(det_jacobian))
        result = (self.W((v-v.T+torch.eye(self.n)))).sum()/(self.n**2-self.n)/2 + (self.D * sum_of_log_jacobians).mean()
        
        return result




