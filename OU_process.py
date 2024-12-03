import numpy as np



# 1-dimensional OU process 
# dX_t = -theta * (X_t - mu) dt + sigma dW_t

class OU_parameters(object):
    def __init__(self):
        self.theta = 1
        self.sigma =  np.sqrt(2)
        self.mu_ou = 0
    def parameters(self):
        return self.theta, self.sigma, self.mu_ou
    


## OU process with initial condition being a Gaussian 
class OU_process(object):
    def __init__(self, theta, sigma, mu):
        self.theta = theta
        self.sigma = sigma
        self.D = sigma**2/2
        self.mu = mu

    def density(self,x,t):
        a = self.theta/2/self.D/(1-np.exp(-2*self.theta*t))
        c = np.exp(-self.theta*t)
        den = 2/np.sqrt(2*np.pi)*np.sqrt(self.theta/(2*np.pi*self.D*(1-np.exp(-2*self.theta*t))))\
            * np.sqrt(np.pi)*np.exp(-(a*(x - self.mu*(1-np.exp(-self.theta *t)))**2)/(2*a*c**2+1))/np.sqrt(4*a*c**2 +2) 
        return den 
    














