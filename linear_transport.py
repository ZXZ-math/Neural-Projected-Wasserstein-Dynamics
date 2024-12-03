import numpy as np 
import matplotlib.pyplot as plt
import torch
from losses import  Linear_loss
import warnings
warnings.filterwarnings("ignore")
from OU_process import OU_process, OU_parameters
from two_layer import two_layer
from compute_G import G_matrix,G_matrix_w

# set seed 
seed_num = 1099
torch.manual_seed(seed_num) #154 #1099
np.random.seed(0)


# number of particles 
num_particles = int(1e6)
num_particles_final = int(1e6)

# currently only dim = 1
dimension = 1

############################################################
# OU process parameters
############################################################
# dX_t = -theta * (X_t - mu) dt + sigma dW_t
OU = OU_parameters()
theta_ou,sigma_ou,mu_ou  = OU.parameters()

############################################################
# choice of potential
############################################################
# 1: quadratic. 2: sixth order polynomial. 3: quartic 
choice = 3
def V(x,dimension = dimension):
    if choice == 1:
        return theta_ou * (x-mu_ou)**2/2
    if choice == 2:
        return (x-4)**6/6
    if choice == 3:
        return (x-1)**4/4 -(x-1)**2/2
        
def grad_V(x,dimension = dimension):
    if choice == 1:
        return theta_ou*(x-mu_ou)
    if choice ==2:
        return (x-4)**5
    if choice == 3:
        return (x-1)**3 -(x-1) 



############################################################
# reference distribution: Gaussian
############################################################
ref_mean, ref_var = 0.0, 1

def sample_X(num_particles=num_particles):
    weights = torch.zeros((num_particles,dimension))
    ## Gaussian
    X = torch.normal(ref_mean, std = np.sqrt(ref_var), size=(num_particles,dimension))
    X_final = torch.normal(ref_mean, std = np.sqrt(ref_var), size=(num_particles_final,dimension))
    return X,X_final,weights

X,X_final,weights = sample_X()
weights= weights.to(dtype=torch.float64)/np.sqrt(np.pi)

# pdf of standard gaussian
def gauss(x,mean=ref_mean,var=ref_var,dim=1): 
    gauss = 1/(2*np.pi)**(dim/2)/np.sqrt(var)*np.exp(-(x-mean)**2/2/var) 
    return gauss

def initial_dist(x):
    return gauss(x)
   

############################################################
# solution of the linear transport equation
############################################################
# dp/dt = p * V'' + dp/dx * V' where V(x) = theta_ou * (x-mu_ou)**2/2
# and p(x,0) = gauss(x)
def linear_pde(x,t,mu=mu_ou,theta=theta_ou):
    arg = (x - mu)*np.exp(t*theta)+mu
    return gauss(arg)*np.exp(t*theta)

def safe_log(z):
    return torch.log(z + 1e-7)

linear_loss = Linear_loss(potential=V,dimension=dimension,weights=weights)
interval = 200 # save every this many steps 
m_list = np.array([4,8]) # number of params of neural network is 4 * m_list[i] 


if choice == 1:
    lr =1e-3
    final_T = 1
elif choice == 2:
    lr = 1e-6
    final_T = 1e-3
else:
    lr = 2e-4
    final_T = 2e-1 
Max_Iter = int(np.ceil(final_T/lr)) + 1
x_size = int(1e6)
x_axis = np.linspace(-6,6,x_size).reshape(x_size,1)
y_axis_final = np.zeros((len(m_list),np.shape(x_axis)[0]))
use_full_G = True # set True if update both weights and bias; False when only update weights
bias_init = 4 # bias initialization bound for the neural network: [-bias_init, bias_init]
    
for index,m in enumerate(m_list):
    ############################################################
    # optimization parameters
    ############################################################
    Max_Iter = int(np.ceil(final_T/lr)) + 1
    fully_con = two_layer(m = m,bias_init=bias_init)
    # lr adjusted by m^2, due to our initialization 
    optimizer = torch.optim.SGD(fully_con.parameters(),lr = lr *(m)**2 ) 
    
    error = np.zeros(Max_Iter)

    for k in range(int(np.ceil(final_T/lr))+1 ):
        optimizer.zero_grad()

        zk = fully_con(X) 

        my_loss = linear_loss(zk)
       
        my_loss.backward(retain_graph=True) 

        if use_full_G: # update both weights and bias 
            agg_gram = G_matrix(w = fully_con.w.data[0:m],wtilde=fully_con.w.data[m:],b=fully_con.bias.data[0:m],btilde=fully_con.bias.data[m:],m=m)
            old_grad_w = fully_con.w.grad.data
            old_grad_b = fully_con.bias.grad.data
            old_grad = torch.cat((old_grad_b,old_grad_w))
            new_grad = torch.linalg.lstsq(agg_gram,old_grad).solution
            fully_con.bias.grad = new_grad[:2*m]
            fully_con.w.grad = new_grad[m*2:]

        else:
            #only update weights
            agg_gram = G_matrix_w(b=fully_con.bias.data[0:m],btilde=fully_con.bias.data[m:],m=m)
            old_grad_w = fully_con.w.grad.data
            old_grad = old_grad_w
            new_grad = torch.linalg.lstsq(agg_gram,old_grad).solution
            fully_con.w.grad = new_grad
            fully_con.bias.grad = torch.zeros(2*m)

        if k%interval ==0:
            print(f'm={m}, outer iteration: {k}/{int(np.ceil(final_T/lr))}, outer loss: {my_loss}' )
  
        optimizer.step()
    y_axis_final[index,:] = fully_con(torch.tensor(x_axis)).detach().numpy().reshape(-1)


    
num_error = len(m_list)
err_w = np.zeros(num_error)

try:
    y_axis_final = y_axis_final.detach().numpy()
    x_axis = x_axis.numpy()
except:
    y_axis_final = y_axis_final
    x_axis = x_axis
t = final_T +lr * 2
def true_y(t,x_axis):
    if choice == 1:
        return mu_ou + np.exp(-t)*(x_axis-mu_ou)
    if choice == 2:
        return (1/(x_axis-4)**4 + 4*t)**(-1/4)*np.sign(x_axis-4)+4
    if choice == 3:
        return np.exp(t)/(np.sqrt(1/(x_axis-1)**2 + np.exp(2*t)-1))*np.sign(x_axis-1)+1
for i in range(num_error):
    err_w[i] = (np.dot(initial_dist(x_axis).reshape(-1,),abs(y_axis_final[i]-true_y(t,x_axis).reshape(-1,))**1)*(x_axis[1]-x_axis[0]))**(1/1)
plt.figure(1)
plt.subplot(121)
plt.xlabel('log N')
plt.ylabel('log error')
plt.plot(np.log10(m_list),np.log10(err_w),'b*-')
#plt.savefig("W1_x_power6_fullG_initb4.pdf", format="pdf", bbox_inches="tight")
plt.subplot(122)
plt.plot(x_axis,true_y(t,x_axis),label='analytic')
plt.plot(x_axis,y_axis_final[-1],label='computed')
plt.legend()
plt.show()