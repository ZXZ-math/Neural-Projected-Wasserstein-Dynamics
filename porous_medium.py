import numpy as np 
import matplotlib.pyplot as plt
import torch
from losses import Porous_medium_loss
import warnings
warnings.filterwarnings("ignore")
from two_layer import two_layer
from densities import Porous_medium
from compute_G import G_matrix,G_matrix_w
from FPE import transport_map
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
# reference distribution: porous media
############################################################
ref_dist = 1
ref_mean, ref_var = 0.0, 1
unif_bound = 1
lam = 1
t0=1
C = (np.sqrt(3)/8)**(2/3)
x_upper = np.sqrt(12 * C *t0**(2/3))
x_lower = -x_upper
po = Porous_medium(t0=t0)

def sample_X(ref_dist,num_particles=num_particles):
    weights = torch.zeros((num_particles,dimension))
    X = torch.tensor(po.sampling(num_particles=num_particles))
    X_final = torch.tensor(po.sampling(num_particles=num_particles_final))
    X = X.view((num_particles,dimension))
    X_final = X_final.view((num_particles_final,dimension))
    return X,X_final,weights

X,X_final,weights = sample_X(ref_dist=ref_dist)
weights= weights.to(dtype=torch.float64)/np.sqrt(np.pi)

# pdf of porous medium 
def porous(x):
    return po.density(x=x,t=0)

def initial_dist(x):
    return porous(x)



porous_medium_loss = Porous_medium_loss(m=2)
interval = 200 # save every this many steps 

m_list = np.array([4])


lr =1e-3
final_T = 1
Max_Iter = int(np.ceil(final_T/lr)) + 1
output_history_list = torch.zeros([len(m_list),int((Max_Iter-1)/interval)+1,num_particles_final])
bias_init = 3**(2/3) * t0**(1/3) 
x_size = int(1e6) 
x_axis = np.linspace(-3**(2/3) * 2**(1/3),3**(2/3) * 2**(1/3),x_size).reshape(x_size,1)
y_axis_final = np.zeros((len(m_list),np.shape(x_axis)[0]))
det_jac_final = np.zeros((len(m_list),np.shape(x_axis)[0]))
use_full_G = True 
eps = 2e-6
    
for index,m in enumerate(m_list):
    ############################################################
    # optimization parameters
    ############################################################
    Max_Iter = int(np.ceil(final_T/lr)) + 1
    output_history_list = torch.zeros([len(m_list),int((Max_Iter-1)/interval)+1,num_particles_final])
    y_axis = np.zeros((int((Max_Iter-1)/interval)+1,np.shape(x_axis)[0]))
    fully_con = two_layer(m = m,bias_init=bias_init)
    init_fc = two_layer(m = m,bias_init=bias_init)
    init_fc.load_state_dict(fully_con.state_dict())
    optimizer = torch.optim.SGD(fully_con.parameters(),lr = lr*(m)**2 )
    
    
    error = np.zeros(Max_Iter)
    out_history_ = torch.zeros([int((Max_Iter-1)/interval)+1,num_particles_final])
    b_count = 0

    for k in tqdm(range(int(np.ceil(final_T/lr))+1 )):
        optimizer.zero_grad()
        if k%interval ==0:
            out = fully_con(X_final)
            out = out.view(-1)
            out_history_[int(k/interval),:] = out
            y_axis[int(k/interval),:] = fully_con(torch.tensor(x_axis)).detach().numpy().reshape(-1)
            
        zk = fully_con(X) 
        

        det_jac = fully_con.det_jac(X)
        pz = po.density(x=X,t=0).view(num_particles)
        my_loss = porous_medium_loss(det_jacobian=det_jac,pz=pz)
        
        my_loss.backward(retain_graph=True) 


        if use_full_G:
            agg_gram = G_matrix(w = fully_con.w.data[0:m],wtilde=fully_con.w.data[m:],b=fully_con.bias.data[0:m],btilde=fully_con.bias.data[m:],m=m)
   
            old_grad_w = fully_con.w.grad.data      
    
            old_grad_b = torch.zeros(2*m)
            old_grad_b[0:m] =  initial_dist(fully_con.bias.data[0:m])**2\
                *(1/fully_con.det_jac(fully_con.bias.data[0:m].view(-1,1)-eps )-1/fully_con.det_jac(fully_con.bias.data[0:m].view(-1,1)))
            old_grad_b[m:] =  -initial_dist(fully_con.bias.data[m:])**2\
                *(1/fully_con.det_jac(fully_con.bias.data[m:].view(-1,1) +eps)-1/fully_con.det_jac(fully_con.bias.data[m:].view(-1,1)))
            
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
T = lr*(Max_Iter-1)
num_plots = 6
t = np.linspace(0,T,num_plots)
t[0]=1e-10
solution = po.density

num_bins = 100
plt.figure(1)
for i,tau in enumerate(t):
    ax = plt.subplot(3, 2, i+1)
    if i == 0:
        out = init_fc(X_final).detach().numpy()
        n1,bins1, patches1 = ax.hist(out, bins = num_bins,density=True)
    else:
        n1,bins1, patches1 = ax.hist(out_history[round(tau/lr/interval),:].detach().numpy(), bins = num_bins,density=True)

    bins1_midpoint = np.array([(bins1[j]+bins1[j+1])/2 for j in range(1,num_bins)])
    
    xx = np.linspace(bins1[0]-0.5,bins1[-1]+0.5,10000)
    yy = solution(x=xx,t=tau)
    
    plt.plot(xx,yy,label='analytic')
    plt.title(f't={"{:.2f}".format(tau)}')
plt.tight_layout()



x_axis = x_axis.reshape(x_size)
rho_0 = initial_dist(x_axis)
dt = lr/20
x_diff = np.array([x_axis[1]-x_axis[0]]+[x_axis[i+1]-x_axis[i] for i in range(x_size-1)])
my_sol = po.density(x=x_axis,t=final_T+lr*2)
my_sol_cdf  = np.zeros(x_size)
my_sol_cdf[0] = 0
for ii in range(1,x_size):
    my_sol_cdf[ii] = my_sol_cdf[ii-1] + my_sol[ii]*x_diff[ii]

t_map = transport_map(xx=x_axis,my_sol_cdf=my_sol_cdf,new_xx=x_axis,pm=True)
upper_bound = 3
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
    print(i)
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