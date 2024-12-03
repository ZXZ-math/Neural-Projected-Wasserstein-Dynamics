import numpy as np
import torch

class Porous_medium(object):
    def __init__(self,t0,b=(np.sqrt(3)/8)**(2/3),m=2):
        self.m = m
        self.t0 = t0
        self.alpha = 1/(self.m+1)
        self.beta = 1/(self.m+1)
        self.b = b

    # def unnormalized_density(self,x,t):
    #     arg = self.b- (self.m-1)/(2*self.m)*self.beta*x**2/(t+self.t0)**(2*self.beta)
    #     arg = np.maximum(arg,0)
    #     den = 1/(t+self.t0)**(self.alpha)*arg**(1/(self.m-1))
    #     return den 
    
    # def density(self,x,t): 
    #     ## only coded for the case m = 2
    #     x_pos = np.sqrt((t+self.t0)**(2*self.beta)*self.b /self.beta * 2*self.m/(self.m-1))
    #     x_neg = - x_pos
    #     if self.m == 2:
    #         integral = self.b*(x_pos-x_neg)-1/12/(t+self.t0)**(2/3)*1/3*(x_pos**3-x_neg**3)
    #         integral = integral/((t+self.t0)**(1/3))
    #         den = self.unnormalized_density(x,t)/integral
    #     return den 
    
    def density(self,x,t):
        arg = self.b- (self.m-1)/(2*self.m)*self.beta*x**2/(t+self.t0)**(2*self.beta)
        arg = np.maximum(arg,0)
        den = 1/(t+self.t0)**(self.alpha)*arg**(1/(self.m-1))
        return den 
    
    def gauss_pdf(self,x,var):
        gauss = 1/(2*np.pi)**(1/2)/np.sqrt(var)*np.exp(-(x)**2/2/var) # pdf of a standard gaussian dist 
        return gauss

    ## sample the reference measure at t0 using rejection sampling
    def sampling(self,num_particles,batch=1000):
        x_pos = np.sqrt((self.t0)**(2*self.beta)*self.b /self.beta * 2*self.m/(self.m-1))
        var = (x_pos/2)**2
        samples = []
        while len(samples) < num_particles:
            X = np.random.normal(0,x_pos/2,size=batch)
            u = np.random.rand(batch)
            factor = 5
            samples += X[u <= self.density(x=X,t=0)/(factor*self.gauss_pdf(x=X,var=var)) ].tolist()
        return samples[:num_particles]







