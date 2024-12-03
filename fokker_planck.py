import numpy as np 
import matplotlib.pyplot as plt
import torch
from losses import Relative_Entropy_FC
import warnings
warnings.filterwarnings("ignore")
from OU_process import OU_process, OU_parameters
from two_layer import two_layer
from compute_G import G_matrix,G_matrix_w
from FPE import FPE, transport_map
from tqdm import tqdm 


# set seed 
seed_num = 1099
torch.manual_seed(seed_num) #154 #1099
np.random.seed(0)


# number of particles 
num_particles = int(1e6)
num_particles_final = int(1e6)
Y_size = int(num_particles)

# currently only dim = 1
dimension = 1

############################################################
# OU process parameters
############################################################
# dX_t = -theta * (X_t - mu) dt + sigma dW_t
OU = OU_parameters()
theta_ou,sigma_ou,mu_ou  = OU.parameters()
D = sigma_ou**2/2

############################################################
# choice of potential
############################################################
# 1: quadratic. 2: sixth order polynomial. 3: quartic 
choice = 2
def V(x,dimension = dimension):
    if choice == 1:
        # quadratic，this is also an OU process 
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


relative_entropy_loss = Relative_Entropy_FC(potential = V, dimension = dimension,D=D,weights=weights)
interval = 200 # save every this many steps 



if choice == 1:
    lr = 1e-3
    final_T = 1
    use_full_G = False 
    m_list = np.array([8])

elif choice == 2:
    lr = 1e-6
    final_T = 1e-3
    m_list = np.array([16])
    use_full_G = True 

else:
    lr = 2e-4
    final_T = 2e-1 
    m_list = np.array([16])
    use_full_G = True 


Max_Iter = int(np.ceil(final_T/lr)) + 1
output_history_list = torch.zeros([len(m_list),int((Max_Iter-1)/interval)+1,num_particles_final])
x_size = int(1e6)
x_axis = np.linspace(-6,6,x_size).reshape(x_size,1)
y_axis_final = np.zeros((len(m_list),np.shape(x_axis)[0]))
det_jac_final = np.zeros((len(m_list),np.shape(x_axis)[0]))
bias_init = 4
eps = 2e-6
    
for index,m in enumerate(m_list):
    ############################################################
    # optimization parameters
    ############################################################
    Max_Iter = int(np.ceil(final_T/lr)) + 1
    output_history_list = torch.zeros([len(m_list),int((Max_Iter-1)/interval)+1,num_particles_final])
    y_axis = np.zeros((int((Max_Iter-1)/interval)+1,np.shape(x_axis)[0]))
    if choice == 1:
        fully_con = two_layer(m = m,bias_init=bias_init,scale=False)
        init_fc = two_layer(m = m,bias_init=bias_init,scale=False)
        optimizer = torch.optim.SGD(fully_con.parameters(),lr = lr)
    else:
        fully_con = two_layer(m = m,bias_init=bias_init)
        init_fc = two_layer(m = m,bias_init=bias_init)
        optimizer = torch.optim.SGD(fully_con.parameters(),lr = lr*(m)**2)
    init_fc.load_state_dict(fully_con.state_dict())
   
    
    
    error = np.zeros(Max_Iter)
    out_history_ = torch.zeros([int((Max_Iter-1)/interval)+1,num_particles_final])

    for k in tqdm(range(int(np.ceil(final_T/lr))+1 )):
        optimizer.zero_grad()
        if k%interval ==0:
            out = fully_con(X_final)
            out = out.view(-1)
            out_history_[int(k/interval),:] = out
            y_axis[int(k/interval),:] = fully_con(torch.tensor(x_axis)).detach().numpy().reshape(-1)
            
        zk = fully_con(X) 
        

        det_jac= fully_con.det_jac(X)
        my_loss = relative_entropy_loss(zk=zk,det_jacobian=det_jac)
       
        my_loss.backward(retain_graph=True) 


        if use_full_G:

            agg_gram = G_matrix(w = fully_con.w.data[0:m],wtilde=fully_con.w.data[m:],b=fully_con.bias.data[0:m],btilde=fully_con.bias.data[m:],m=m)

            
            old_grad_w = fully_con.w.grad.data
            

            old_grad_b = torch.zeros(2*m)

            old_grad_b[0:m] = fully_con.bias.grad.data[0:m] + initial_dist(fully_con.bias.data[0:m])\
                *torch.log(fully_con.det_jac(fully_con.bias.data[0:m].view(-1,1)  )/fully_con.det_jac(fully_con.bias.data[0:m].view(-1,1) - eps)) 
            
            old_grad_b[m:] = fully_con.bias.grad.data[m:] - initial_dist(fully_con.bias.data[m:])\
                *torch.log(fully_con.det_jac(fully_con.bias.data[m:].view(-1,1)  )/fully_con.det_jac(fully_con.bias.data[m:].view(-1,1) + eps ))  
            
            
            old_grad = torch.cat((old_grad_b,old_grad_w))
            new_grad = torch.linalg.lstsq(agg_gram,old_grad).solution
            fully_con.bias.grad = new_grad[:2*m]
            fully_con.w.grad = new_grad[m*2:]


        else:
            #only update w
            agg_gram = G_matrix_w(b=fully_con.bias.data[0:m],btilde=fully_con.bias.data[m:],m=m)
            
            old_grad_w = fully_con.w.grad.data
            old_grad = old_grad_w
            new_grad = torch.linalg.lstsq(agg_gram,old_grad).solution
        
            fully_con.w.grad = new_grad
            fully_con.bias.grad = torch.zeros(2*m)

        if k%interval ==0:
            print(f'm={m}, outer iteration: {k}/{int(np.ceil(final_T/lr))}, outer loss: {my_loss}' )
    
        optimizer.step()


    if out_history_[0,0] != 0:    
        out_history = out_history_
    output_history_list[index,:,:] = out_history_
    y_axis_final[index,:] = fully_con(torch.tensor(x_axis)).detach().numpy().reshape(-1)
    det_jac_final[index,:] = fully_con.det_jac(torch.tensor(x_axis)).detach().numpy().reshape(-1)

############################################################
# plot density evolution 
############################################################
print('plotting...')
T = lr*(Max_Iter-1)
num_plots = 6
t = np.linspace(0,T,num_plots)
t[0] = 1e-10
num_bins = 100

if choice == 1:
    my_ou = OU_process(theta = theta_ou, sigma = sigma_ou, mu = mu_ou)
    solution = my_ou.density
else:
    x_axis = x_axis.reshape(x_size)
    rho_0 = initial_dist(x_axis)
    dt = lr/2
    my_FPE_arr = FPE(grad_potential=grad_V,T = final_T+lr,dt = dt,rho_0 =rho_0,x=x_axis,n=num_plots)
    my_sol_arr = my_FPE_arr.solution()    
plt.figure(1)
for i,tau in enumerate(t):
    ax = plt.subplot(3, 2, i+1)
    if i == 0:
        out = init_fc(X_final).detach().numpy()
        n1,bins1, patches1 = ax.hist(out, bins = num_bins,density=True)
    else:
        n1,bins1, patches1 = ax.hist(out_history[round(tau/lr/interval),:].detach().numpy(), bins = num_bins,density=True)


    bins1_midpoint = np.array([(bins1[j]+bins1[j+1])/2 for j in range(1,num_bins)])

    if choice == 1:
        xx = np.linspace(bins1[0]-0.5,bins1[-1]+0.5,10000)
        yy = solution(x=xx,t=tau)
    else:
        indices = np.where((x_axis<=bins1[-1]+0.5)&(x_axis>=bins1[0]-0.5))
        xx = x_axis[indices]
        yy = my_sol_arr[:,i]
        yy = yy[indices]
    plt.plot(xx,yy,label='analytic')
    plt.title(f't={"{:.3f}".format(tau)}')
plt.tight_layout()


############################################################
# plot transport map and error 
############################################################


x_axis = x_axis.reshape(x_size)
rho_0 = initial_dist(x_axis)
dt = lr/2
x_diff = np.array([x_axis[1]-x_axis[0]]+[x_axis[i+1]-x_axis[i] for i in range(x_size-1)])
if choice == 1:
    my_ou_density = OU_process(theta = theta_ou, sigma = sigma_ou, mu = mu_ou).density
    my_sol = my_ou_density(x=x_axis,t=final_T+lr*1)
else:
    my_FPE = FPE(grad_potential=grad_V,T = final_T+1*lr,dt = dt,rho_0 =rho_0,x=x_axis)
    my_sol = my_FPE.solution()

my_sol_cdf  = np.zeros(x_size)
my_sol_cdf[0] = 0
for ii in range(1,x_size):
    my_sol_cdf[ii] = my_sol_cdf[ii-1] + my_sol[ii]*x_diff[ii]
if choice == 1:
    correction = 1
    t_map = mu_ou*(1-np.exp(-(final_T+lr*correction)*theta_ou)) + (np.exp(-2*theta_ou*(final_T+lr*correction))+D*(1-np.exp(-2*(final_T+lr*correction)*theta_ou))/theta_ou)**(1/2)*x_axis
else:
    t_map = transport_map(xx=x_axis,my_sol_cdf=my_sol_cdf,new_xx=x_axis)
upper_bound = 5.5
xx_axis = x_axis[(np.abs(x_axis)<=upper_bound).nonzero()]
tt_map = t_map[(np.abs(x_axis)<=upper_bound).nonzero()]
err_w = np.zeros(len(m_list))
try:
    y_axis_final = y_axis_final.detach().numpy()
except:
    y_axis_final = y_axis_final
plt.figure(2)
plt.subplot(1,2,1)
dx = xx_axis[1]-xx_axis[0]
for i in range(len(m_list)): 
    yy = y_axis_final[i,:]
    yy = yy.reshape(-1)
    yy_axis = yy[(np.abs(x_axis)<=upper_bound).nonzero()]
    err_w[i] = (((np.abs(yy_axis-tt_map))**1*initial_dist(xx_axis)* dx).sum())**(1/1)
plt.plot(np.log10(m_list),np.log10(err_w),'b*-')

# transport map 
plt.subplot(1,2,2)
plt.plot(xx_axis,tt_map,label='analytic')

plt.plot(xx_axis,yy_axis,label='computed')

plt.legend()

plt.show()