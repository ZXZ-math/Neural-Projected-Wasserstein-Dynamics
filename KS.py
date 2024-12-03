import numpy as np 
import matplotlib.pyplot as plt
import torch
from losses import Interaction_loss
import warnings
warnings.filterwarnings("ignore")
from two_layer import two_layer
from compute_G import G_matrix,G_matrix_w
from tqdm import tqdm 


# set seed 
seed_num = 42
torch.manual_seed(seed_num) 
np.random.seed(0)


# number of particles 
num_particles = int(2000)
num_particles_final = int(1e7)
Y_size = int(num_particles)

# currently only dim = 1
dimension = 1



############################################################
# Interaction potential
############################################################
chi = .5 # 0.5 or 1.5 
def W(x):
    return torch.log(torch.abs(x))*2*chi

############################################################
# reference distribution: Gaussian, uniform, exponential, porous media
############################################################
ref_mean, ref_var = 0.0, 1


def sample_X(num_particles=num_particles):
    weights = torch.zeros((num_particles,dimension))
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
   


interaction_loss = Interaction_loss(W=W,n=num_particles)
interval = 100 # save every this many steps 
m_list = np.array([32])

lr =3e-4
final_T = 3e-1
Max_Iter = int(np.ceil(final_T/lr)) + 1
output_history_list = torch.zeros([len(m_list),int((Max_Iter-1)/interval)+1,num_particles_final])
x_size = int(5e6)
x_axis = np.linspace(-8,8,x_size).reshape(x_size,1)
y_axis_final = np.zeros((len(m_list),np.shape(x_axis)[0]))
det_jac_final = np.zeros((len(m_list),np.shape(x_axis)[0]))
bias_init = 4
use_full_G = True 
eps = 2e-6
    
for index,m in enumerate(m_list):
    ############################################################
    # optimization parameters
    ############################################################
    Max_Iter = int(np.ceil(final_T/lr)) + 1
    output_history_list = torch.zeros([len(m_list),int((Max_Iter-1)/interval)+1,num_particles_final])
    y_axis = np.zeros((int((Max_Iter-1)/interval)+1,np.shape(x_axis)[0]))
    fully_con = two_layer(m = m,bias_init=bias_init,scale=False)
    optimizer = torch.optim.SGD(fully_con.parameters(),lr = lr)

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
        my_loss = interaction_loss(zk.view(-1,1),det_jacobian=det_jac)
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





T = lr*(Max_Iter-1)
num_plots = 6
t = np.linspace(0,T,num_plots)
t[0]=1e-10
num_bins = 100
plt.figure(1)
for i,tau in enumerate(t):
    ax = plt.subplot(3, 2, i+1)
    if i == 0:
        out = (X_final).detach().numpy()
        n1,bins1, patches1 = ax.hist(out, bins = num_bins,density=True)
    else:
        n1,bins1, patches1 = ax.hist(out_history[round(tau/lr/interval),:].detach().numpy(), bins = num_bins,density=True)

    bins1_midpoint = np.array([(bins1[j]+bins1[j+1])/2 for j in range(1,num_bins)])
    

    plt.title(f't={"{:.2f}".format(tau)}')
plt.tight_layout()

try:
     output_history_list.detach().numpy()
except:
     output_history_list = output_history_list

L = np.shape(output_history_list)[1]
second_moment = np.zeros(L)
analytic_second_moment = np.zeros(L)
for i in range(L):
     second_moment[i] = (output_history_list[0,i,:]**2).mean()
     real_t = i*lr*interval
     analytic_second_moment[i] = 2*(1-chi)*real_t + 1
t = np.array([i*lr*interval for i in range(L)])
plt.figure(2)
plt.subplot(1,2,1)
plt.plot(t,analytic_second_moment,'ro-',label='analytic')
plt.plot(t,second_moment,'b*-.',label='numerical')
plt.legend()




x_axis = x_axis.reshape(x_size)
upper_bound = 8
xx_axis = x_axis[(np.abs(x_axis)<=upper_bound).nonzero()]

try:
    y_axis_final = y_axis_final.detach().numpy()
except:
    y_axis_final = y_axis_final
#axs = plt.subplot(2,1,1)
# axss = plt.subplot(1,1,1)
# dx = xx_axis[1]-xx_axis[0]
for i in range(len(m_list)): 
    yy = y_axis_final[i,:]
    yy = yy.reshape(-1)
    yy_axis = yy[(np.abs(x_axis)<=upper_bound).nonzero()]
    





# transport map 
plt.subplot(1,2,2)

plt.plot(xx_axis,yy_axis,label='computed')

plt.legend()
plt.show()


