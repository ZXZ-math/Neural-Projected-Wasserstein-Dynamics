import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm

## computes the solution to FPE 
# d(rho)/dt + d/dx (rho(t,x)*grad_V(x)) + d^2/dx^2 rho(t,x) = 0
# with initial data rho(0,x) = rho_0(x) and vanishing BC 
# forward in space, backward in time 

class FPE(object):
    def __init__(self,grad_potential,T,dt,rho_0,x,dx=0,n=2):
        self.grad_V = grad_potential
        self.T = T
        self.dt = dt 
        self.dx = dx
        self.x = x
        self.rho_0 = rho_0 .reshape(-1,1)
        self.size = len(rho_0)
        self.n = n
        self.interval = int(T/dt/(n-1))
    def solution(self):
        M = np.shape(self.x)[0]
        max_iter = round(self.T/self.dt)
        
        if self.dx != 0:
            above_diag = -self.dt/self.dx*self.grad_V(self.x)-self.dt/self.dx**2
            above_diag[0] = 0
            diagonal = self.dt/self.dx*self.grad_V(self.x) + 1 + 2 * self.dt/self.dx**2
            below_diag = -self.dt/self.dx**2 * np.ones((M))
            below_diag[-1] = 0
        else:
            x_diff = np.array([self.x[i+1]-self.x[i] for i in range(M-1)])
            x_diff = np.append(np.array([x_diff[0]]),x_diff)
            x_diff = np.append(x_diff,np.array([x_diff[-1]]))
            lap_above_diag = np.array([0]+[2/(x_diff[i+1]*(x_diff[i+1]+x_diff[i])) for i in range(M-1)])
            lap_diag = np.array([-2/x_diff[i+1]/x_diff[i] for i in range(M)])
            lap_below_diag = np.array([2/(x_diff[i]*(x_diff[i+1]+x_diff[i])) for i in range(1,M)]+[0])
            grad_diag = -self.grad_V(self.x)/x_diff[1:]
            grad_above_diag = np.array([0]+[self.grad_V(self.x[i+1])/x_diff[i+1] for i in range(M-1)])
            diagonal = -self.dt * (lap_diag+grad_diag)+1
            below_diag = -self.dt * lap_below_diag 
            above_diag = -self.dt * (lap_above_diag + grad_above_diag)

        ab = np.array([above_diag, diagonal, below_diag])
        rho_arr = np.zeros((self.size,(self.n)))
        rho_arr[:,0] = self.rho_0.reshape(-1,)
        #print(f'interval: {self.interval}')
        for i in range(max_iter):
            self.rho_0 = solve_banded((1,1),ab,self.rho_0)
            if (i+1)%self.interval==0:
                rho_arr[:,int((i+1)/self.interval)] = self.rho_0.reshape(-1,)
                #print(f'max_iter: {max_iter}, index:{int((i+1)/self.interval)}, i: {i}' )
        if self.n==2:
            return rho_arr[:,-1]
        else:
            return rho_arr


    

def find_index(z,my_sol_cdf,starting_index):
    if my_sol_cdf[starting_index]>z:
        return starting_index
    for i in range(starting_index,len(my_sol_cdf)-1):
        if my_sol_cdf[i+1]>z and my_sol_cdf[i]<=z:
            break
    return i


def transport_map(xx,my_sol_cdf,new_xx,pm=False):
    if pm:
        t0=1
        C = (np.sqrt(3)/8)**(2/3)
        x_upper = np.sqrt(12 * C *t0**(2/3))
        x_lower = -x_upper
        upper_bound = np.minimum(xx,x_upper)
        ref_cdf = (xx>x_lower)*t0**(-1/3)*(C*(upper_bound-x_lower)-(upper_bound**3-x_lower**3)*t0**(-2/3)/36)
    else:
        ref_cdf = norm.cdf(xx)
    
    t_map = np.zeros(len(xx))
    starting_index = 0
    for (i,z) in enumerate(ref_cdf):
        starting_index = find_index(z=z,my_sol_cdf=my_sol_cdf,starting_index=starting_index)
        if starting_index == 0:
            t_map[i] = new_xx[starting_index]
        else:
            ratio = (z-my_sol_cdf[starting_index])/(my_sol_cdf[starting_index+1]-my_sol_cdf[starting_index])
            if ratio<1:
                t_map[i] = new_xx[starting_index] + ratio*(new_xx[starting_index+1]-new_xx[starting_index])
            else:
                t_map[i] = new_xx[starting_index] + 1/2*(new_xx[starting_index+1]-new_xx[starting_index])
    return t_map
