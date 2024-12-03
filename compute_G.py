# Returns the metric tensor G with a specific reference measure.

import torch
import numpy as np
from scipy import special

##############################################################################################################################
# Base measure
##############################################################################################################################


#############################################################################################
#############################################################################################
# Guassian (Use this for transport eqn, fokker-planck eqn and KS eqn.)
#############################################################################################
#############################################################################################

def x_square_integral(l,u):
    # computes integral of x**2 p(x) dx where p(x) is the pdf of a standard gaussian
    # from l to u
    if l >= u:
        return 0
    else:
        if u == torch.inf:
            upper = np.sqrt(torch.pi/2) #*torch.erf(u/torch.sqrt(2))    
        else:
            upper = np.sqrt(torch.pi/2)*special.erf(u/np.sqrt(2)) - np.exp(-u**2/2)*u    
        if l == -torch.inf: 
            lower = -np.sqrt(np.pi/2) #*torch.erf(l/torch.sqrt(2))
        else:
            lower = np.sqrt(np.pi/2)*special.erf(l/np.sqrt(2)) - np.exp(-l**2/2)*l   
        #print(f'upper:{upper},lower:{lower}')
        return (upper-lower)/np.sqrt(2*np.pi)

def x_integral(l,u): 
    # computes integral of x p(x) dx where p(x) is the pdf of a standard gaussian
    # from l to u
    if l >= u:
        return 0
    else:
        upper = - np.exp(-u**2/2)
        lower = - np.exp(-l**2/2)
        return (upper-lower)/np.sqrt(2*np.pi)

def integral(l,u):
    # computes integral of p(x) dx where p(x) is the pdf of a standard gaussian
    # from l to u
    if l >= u:
        return 0
    else:
        upper = 1/2*(1+special.erf(u/np.sqrt(2)))
        lower = 1/2*(1+special.erf(l/np.sqrt(2)))
        return upper-lower

#############################################################################################
#############################################################################################
# Porous Media (Use this for the porous medium eqn)
#############################################################################################
#############################################################################################

# t0 = 1
# C = (np.sqrt(3)/8)**(2/3)
# x_upper = np.sqrt(12 * C *t0**(2/3))
# x_lower = -x_upper

# def x_square_integral(l,u):

#     # computes integral of x**2 p(x) dx where p(x) is the pdf of a porous media
#     # from l to u
#     if l >= u:
#         return 0
#     else:
#         #l = torch.tensor(l)
#         #u = torch.tensor(u)
#         u = np.minimum(u,x_upper)
#         l = np.maximum(l,x_lower)
#         ans = t0**(-1/3)*(C*(u**3-l**3)/3-t0**(-2/3)*(u**5-l**5)/60)
#         return ans

# def x_integral(l,u): 
#    # computes integral of x p(x) dx where p(x) is the pdf of a porous media
#     # from l to u
#     if l >= u:
#         return 0
#     else:
#         #l = torch.tensor(l)
#         #u = torch.tensor(u)
#         u = np.minimum(u,x_upper)
#         l = np.maximum(l,x_lower)
#         ans = t0**(-1/3)*(C*(u**2-l**2)/2-t0**(-2/3)*(u**4-l**4)/48)
#         return ans

# def integral(l,u):
#     # computes integral of p(x) dx where p(x) is the pdf of a porous media
#     # from l to u
#     if l >= u:
#         return 0
#     else:
#         #l = torch.tensor(l)
#         #u = torch.tensor(u)
#         u = np.minimum(u,x_upper)
#         l = np.maximum(l,x_lower)
#         ans = t0**(-1/3)*(C*(u-l)-t0**(-2/3)*(u**3-l**3)/36)
#         return ans

##############################################################################################################################
# building the metric tensor
##############################################################################################################################


def E_ww(b,m):
    # returns an m by m matrix with the i,j entry given by 
    # int_{\max{bi,bj}}^{\infty} (x-bi)(x-bj)p(x) dx 
    # where p(x) is the pdf for a standard gaussian
    M = torch.zeros((m,m))
    for i in range(m):
        for j in range(m):
            bb = torch.max(b[i],b[j])
            term1 = x_square_integral(l=bb,u=torch.inf)
            term2 = -(b[i]+b[j])*x_integral(l=bb,u=torch.inf)
            term3 = b[i]*b[j]*integral(l=bb,u=torch.inf)
            M[i,j] = term1 + term2 + term3
    return M

def E_wtilde_b(w,b,btilde,m):
    # returns an m by m matrix with the i,j entry given by 
    # -wj*int_{bj}^{btilde_i}p(x)(btilde_i - x) dx 
    M = torch.zeros((m,m))
    for k in range(m):
        for l in range(m):
            term1 = btilde[k] * integral(l=b[l],u=btilde[k])
            term2 = -x_integral(l=b[l],u=btilde[k])
            M[k,l] = -w[l]*(term1+term2)
    return M

def E_w_wtilde(b,btilde,m):
    # returns an m by m matrix with the k,l entry given by 
    # int_{bk}^{btilde_l}(x-bk)(btilde_l-x) p(x) dx 
    M = torch.zeros((m,m))
    for k in range(m):
        for l in range(m):
            term1 = -x_square_integral(l=b[k],u=btilde[l])
            term2 = x_integral(l=b[k],u=btilde[l])*(btilde[l]+b[k])
            term3 = -b[k]*btilde[l]*integral(l=b[k],u=btilde[l])
            M[k,l] = term1 + term2 + term3
    return M

def E_bb(w,b,m):
    # returns an m by m matrix with the k,l entry given by 
    # wk * wl (1-F(max(bk,bl))) where F is the cdf
    M = torch.zeros((m,m))
    for k in range(m):
        for l in range(m):
            bb = torch.max(b[k],b[l])
            M[k,l] = w[k]*w[l]*integral(l=bb,u=torch.inf)
    return M

def E_b_btilde(w,wtilde,b,btilde,m):
    # returns an m by m matrix with the k,l entry given by 
    # -wk * wtilde_l * int_{bk}^{btilde_l} p(x) dx 
    M = torch.zeros((m,m))
    for k in range(m):
        for l in range(m):
            M[k,l] = -w[k]*wtilde[l]*integral(l=b[k],u=btilde[l])
    return M

def E_w_b(w,b,m):
    # returns an m by m matrix with the k,l entry given by 
    # -w_l * int_{max(bl,bk)}^{infty} (x-bk) p(x) dx 
    M = torch.zeros((m,m))
    for k in range(m):
        for l in range(m):
            bb = torch.max(b[k],b[l])
            term1 = x_integral(l=bb,u=torch.inf)
            term2 = -b[k] * integral(l=bb,u=torch.inf)
            M[k,l] = -w[l]*(term1+term2)
    return M

def E_w_btilde(wtidle,b,btilde,m):
    # returns an m by m matrix with the k,l entry given by 
    # wtilde_l* int_{bk}^{btilde_l}(x-bk) p(x) dx 
    M = torch.zeros((m,m))
    for k in range(m):
        for l in range(m):
            term1 = x_integral(l=b[k],u=btilde[l])
            term2 = -b[k]*integral(l=b[k],u=btilde[l])
            M[k,l] = wtidle[l]*(term1 + term2)
    return M 

def E_wtilde_btilde(wtilde,btilde,m):
    # returns an m by m matrix with the k,l entry given by 
    # wtilde_l * int_{-infty}^{min(btilde_l,btidle_k)} (btilde_k - x) p(x) dx 
    M = torch.zeros((m,m))
    for k in range(m):
        for l in range(m):
            bb = torch.min(btilde[l],btilde[k])
            term1 = btilde[k]*integral(l=-torch.inf,u=bb)
            term2 = -x_integral(l=-torch.inf,u=bb)
            M[k,l] = wtilde[l]*(term1+term2)
    return M 

def E_wtilde_wtilde(btilde,m):
    # returns an m by m matrix with the k,l entry given by 
    # int_{-infty}^{min(btilde_l,btidle_k)} (btilde_k - x)(btilde_l-x) p(x) dx 
    M = torch.zeros((m,m))
    for k in range(m):
        for l in range(m):
            bb = torch.min(btilde[l],btilde[k])
            term1 = btilde[k]*btilde[l]*integral(l=-torch.inf,u=bb)
            term2 = -(btilde[k]+btilde[l])*x_integral(l=-torch.inf,u=bb)
            term3 = x_square_integral(l=-torch.inf,u=bb)
            M[k,l] = term1 + term2 + term3 
    return M

def E_btilde_btilde(wtilde,btilde,m):
    # returns an m by m matrix with the k,l entry given by 
    # wtilde_k*wtilde_l * int_{-infty}^{min(btilde_l,btidle_k)}  p(x) dx 
    M = torch.zeros((m,m))
    for k in range(m):
        for l in range(m):
            bb = torch.min(btilde[l],btilde[k])
            M[k,l] = wtilde[k]*wtilde[l]*integral(l=-torch.inf,u=bb)
    return M


def G_matrix_b(w,wtilde,b,btilde,m):
    # returns the 2m by 2m metrix matrix for b 
    bb = E_bb(w=w,b=b,m=m)
    b_btilde = E_b_btilde(w=w,wtilde=wtilde,b=b,btilde=btilde,m=m)
    btilde_btilde = E_btilde_btilde(wtilde=wtilde,btilde=btilde,m=m)
    G = torch.zeros((2*m,2*m))
    G[0:m,] = torch.cat([bb,b_btilde],dim=1)
    G[m:,] = torch.cat([b_btilde.T,btilde_btilde],dim = 1)
    return G

def G_matrix_w(b,btilde,m):
    # returns the 2m by 2m metrix matrix for w
    ww = E_ww(b=b,m=m)
    w_wtilde = E_w_wtilde(b=b,btilde=btilde,m=m)
    wtilde_wtilde = E_wtilde_wtilde(btilde=btilde,m=m)
    G = torch.zeros((2*m,2*m))
    G[0:m,] = torch.cat([ww,w_wtilde],dim=1)
    G[m:,] = torch.cat([w_wtilde.T,wtilde_wtilde],dim=1)
    return G



def G_matrix(w,wtilde,b,btilde,m):
    # returns the 4m by 4m metric matrix 
    ww = E_ww(b=b,m=m)
    w_wtilde = E_w_wtilde(b=b,btilde=btilde,m=m)
    bb = E_bb(w=w,b=b,m=m)
    b_btilde = E_b_btilde(w=w,wtilde=wtilde,b=b,btilde=btilde,m=m)
    wb = E_w_b(w=w,b=b,m=m)
    bw = wb.T
    wtilde_b = E_wtilde_b(w=w,b=b,btilde=btilde,m=m)
    b_wtilde = wtilde_b.T
    w_btilde = E_w_btilde(wtidle=wtilde,b=b,btilde=btilde,m=m)
    btilde_w = w_btilde.T
    wtilde_btilde = E_wtilde_btilde(wtilde=wtilde,btilde=btilde,m=m)
    btilde_wtilde = wtilde_btilde.T
    wtilde_wtilde = E_wtilde_wtilde(btilde=btilde,m=m)
    btilde_btilde = E_btilde_btilde(wtilde=wtilde,btilde=btilde,m=m)

    G = torch.zeros((4*m,4*m))
    G[0:m,] = torch.cat([bb,b_btilde,bw,b_wtilde],dim=1)
    G[m:2*m,m:] = torch.cat([btilde_btilde,btilde_w,btilde_wtilde],dim=1)
    G[2*m:3*m,2*m:] = torch.cat([ww,w_wtilde],dim=1)
    G[3*m:,3*m:] = wtilde_wtilde

    G_upper = torch.triu(G,diagonal=1)
    G_upper_diag = torch.triu(G)
    G = G_upper_diag + G_upper.T
    return G 





            




    





