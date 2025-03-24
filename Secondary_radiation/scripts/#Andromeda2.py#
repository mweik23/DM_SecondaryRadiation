import numpy as np
import constants_and_functions as cf
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
import os
from scipy.ndimage import convolve
import time
from sklearn.linear_model import LinearRegression

this_path = os.path.realpath(__file__)
base_path = this_path.split('scripts')[0]
model_func_path = base_path + 'astro_model_funcs/'


B_params = np.load(model_func_path + 'B_params.npy') #[[B0, RB0], [B1, RB1]]

lnrho_interp = np.load(model_func_path + 'lnrho_interp.npy', allow_pickle=True)[()] #ln of interpolation of rho_star

H1_interp = np.load(model_func_path + 'H1_interp.npy', allow_pickle=True)[()] #interpolation of H1 density on disk
H1_params = np.load(model_func_path + 'H1_params.npy') #extrapolation parameters [[norm0, R_sc0], [norm1, R_sc1]]
H1_params2 = np.load(model_func_path + 'H1_params2.npy') #corrected slope and intercept of scale height
H1_coef = np.load(model_func_path + 'H1_coef.npy')[0] #normalization coefficient
H1_bounds = np.load(model_func_path + 'H1_bounds.npy')*1000 #pc

H2_interp = np.load(model_func_path + 'H2_interp.npy', allow_pickle=True)[()] #interpolation of H2 density on disk
H2_params = np.load(model_func_path + 'H2_params.npy') #extrapolation parameters [[norm0, R_sc0], [norm1, R_sc1]]  
H2_hz_params = np.load(model_func_path + 'hz_params_H2.npy')*1000 #corrected scale height paramters [hz_0, R_hz]
H2_coef = np.load(model_func_path + 'H2_coef.npy')[0] #normalization coefficient
H2_bounds = np.load(model_func_path + 'H2_bounds.npy')*1000 #pc

ion_gas_params = np.load(model_func_path + 'ion_gas_params.npy') #ionized gas density parameters [norm, R-scale, z-scale]

#ion gas parameter extraction
nion0 = ion_gas_params[0]
Rion0 = ion_gas_params[1]*1000 #pc
zion0 = ion_gas_params[2]*1000 #pc

#H1 params
hzH1_int = H1_params2[0]*1000 #pc
hzH1_sl = H1_params2[1]

#B params
B0 = B_params[0, 0]
RB0 = B_params[0, 1]*1000 #pc
B1 = B_params[1, 0]
RB1 = B_params[1, 1]*1000 #pc 
zB0 = 4*hzH1_int #pc
zB1 = 4*hzH1_sl

DA = 780000
beta_deg = 77.5
beta = beta_deg*np.pi/180
rat=1/12
D0_bench = 3e28

#magnetic field
#arguments in units of pc
#define astro models                                                                                                                                                                 
def zB(R):
    return zB0+zB1*R
def B(R, z):
    return (B0*np.exp(-R/RB0)+B1*np.exp(-R/RB1))*np.exp(-np.abs(z)/zB(R))

def dBdR(R, z):
    return -((B0/RB0)*np.exp(-R/RB0)+(B1/RB1)*np.exp(-R/RB1))*np.exp(-np.abs(z)/zB(R))+(np.abs(z)/zB(R)**2)*zB1*B(R, z)

#only works for positive z                                                                                                                                                           
def dBdz(R, z):
    return -B(R, z)/zB(R)

#ISRF from starlight
#arguments in units of pc
def rho_star(R, z):
    if np.isscalar(R) and np.isscalar(z):
        res = np.exp(lnrho_interp(R/1000, z/1000))[0]
    elif len(R.shape)>1 and len(z.shape)>1:
        R = R[0]
        z = z[:, 0]
        res = np.exp(lnrho_interp(R/1000, z/1000))
    else:
        res = np.exp(lnrho_interp(R/1000, z/1000))
    return res/500

#H1 density
def H1_extrap(R):
    b0 = R<H1_bounds[0]
    b1 = R>H1_bounds[1]
    return b0*H1_params[0,0]*np.exp(b0*R/(1000*H1_params[0,1])) + H1_interp(R/1000) + b1*H1_params[1,0]*np.exp(-(R*b1)/(1000*H1_params[1,1]))

def n_H1(R, z):
    hz = hzH1_int + hzH1_sl*R
    return H1_coef*np.exp(-np.abs(z)/hz)*H1_extrap(R)


#H2 density
def H2_extrap(R):
    b0 = R<H2_bounds[0]
    b1 = R>H2_bounds[1]
    return b0*H2_params[0, 0]*np.exp(b0*R/(1000*H2_params[0, 1])) + H2_interp(R/1000) + b1*H2_params[1, 0]*np.exp(-(R*b1)/(1000*H2_params[1, 1]))

def hzH2(R, w):
    return w[0]*np.exp(-R/w[1])

epsilon = 0.0001
def n_H2(R, z):
    hz_val = hzH2(R, H2_hz_params)
    return H2_coef*np.exp(-np.abs(z)/(hz_val+epsilon))*H2_extrap(R)

def nH(R, z):
    return n_H2(R, z) + n_H1(R, z)
#ionized gas density
def nion(R, z):
    return nion0*(1/np.cosh(R/Rion0))*(1/np.cosh(z/zion0))

#coefficients                                                                                                                                                                                          
def b(R, z, E):
    bsyn = np.multiply.outer(E**2 ,cf.b0_syn*(B(R, z)**2))
    bIC = np.multiply.outer(E**2, cf.b0_IC*(1 + rho_star(R,z)/cf.rho_CMB))
    bH = np.multiply.outer(E, cf.b0_H*nH(R, z))
    bHe = np.multiply.outer(E, cf.b0_He*rat*nH(R, z))
    bion = np.multiply.outer(E*(1 + np.log(E)/7.94), cf.b0_ion*nion(R, z))
    bC = cf.b0_C*nion(R, z)*(1 + np.log(np.multiply.outer(E, 1/nion(R,z)))/82)
    return bsyn + bIC + bH + bHe + bion+ bC

def D(R, z, E):
    D0_bench = 3e28 #default value                                                                                                                                                                           
    return D0_bench*np.multiply.outer(E**(1/3), (10/B(R, z))**(1/3))         #kpc**2/s                                                                                                                       

def spherical_average_terms(integrands, r, E, D0):
    def get_terms_r(rval):
        if (rval%1000<100):
            print('rval = ', rval)
        integs = [lambda z, func=integrand: func(z, rval, E, D0) for integrand in integrands]
        res = [quad_vec(integ, 0, rval)[0]/rval for integ in integs]
        return res
    results = np.array([get_terms_r(r[i]) for i in range(1, len(r))])
    print(results.shape)
    results = list(np.einsum('kij', results))
# =============================================================================                                                                                                                        
#     for i in range(1, nr+2):                                                                                                                                                                         
#         if i % 100 == 0:                                                                                                                                                                             
#             print('integrating r_i where i = ' + str(i))                                                                                                                                             
#         rval = r[i]                                                                                                                                                                                  
#         integs = [lambda z, func=integrand: func(z, rval, E) for integrand in integrands]                                                                                                            
#         for j in range(len(results)):                                                                                                                                                                
#             results[j][:, i] = quad_vec(integs[j], 0, rval)[0]/rval                                                                                                                                  
# =============================================================================                                                                                                                        
    for j in range(len(results)):
        if j !=2:
            results[j] = np.hstack((integrands[j](0, 0, E, D0).reshape(len(E), 1), results[j]))
    return results

#sph_avgs = [<1/b>, <D/b>, <dDdr/b>]                                                                                                                                                                   
def find_coefficients(sph_avgs, r, E, DM_model, dl=True):
    rr, _ = np.meshgrid(r, E)
    derivs = [d_dx(sph_avgs[i], r) for i in range(2)]
    rho_DM = cf.dist(rr[:, 1:-1], DM_model)
    drhodr = cf.ddistdr(rr[:, 1:-1], DM_model)
    Dbar = np.zeros((len(E), len(r)-1))
    bbar = np.copy(Dbar)
    Dbar[:, 1:] = ((2*drhodr/rho_DM)*sph_avgs[1][:, 1:-1] + derivs[1][:, 1:-1]-sph_avgs[2][:, 0:-1])/((2*drhodr/rho_DM)*sph_avgs[0][:, 1:-1] + derivs[0][:, 1:-1])
    Dbar[:, 0] = sph_avgs[4][:, 0]
    if dl:
        bbar[:, 1:] = sph_avgs[3][:, 1:-1]/sph_avgs[0][:, 1:-1]
    else:
        bbar[:, 1:] = 1/sph_avgs[0][1:-1]
    bbar[:, 0] = b(0, 0, E)
    return bbar, Dbar

def d2_dx2(f, x, axis=-1, lin=True):
    num_axes = len(f.shape)
    if not axis==-1:
        ind = ['i', 'j', 'k', 'l', 'm', 'n']
        ind = ind[:num_axes]
        ind[axis], ind[-1] = ind[-1], ind[axis]
        s = ''.join(ind)
        f = np.einsum(s, f)
    if lin:
        dx = x[1]-x[0]
        sh = tuple(1 if i<num_axes-1 else 3 for i in range(num_axes))
        d2dx2 = (1/(dx**2))*np.array([1, -2, 1])
        d2dx2 = d2dx2.reshape(sh)
        d2fdx2 = convolve(f, d2dx2, mode='constant')
        if not axis==-1:
            d2fdx2 = np.einsum(s, d2fdx2)
    return d2fdx2

def merge_D(Dw_in, Duw_in, r, r0):
    n0 = np.where(np.abs(r-r0)==np.min(np.abs(r-r0)))[0][0]
    print('n0 = ', n0)
    n1 = 2*n0
    n2 = 3*n0
    epsilon = 1e-300
    Dw = np.copy(Dw_in)[:, 1:n2+1]
    Duw = np.copy(Duw_in)[:, 1:n2+1]
    blean = Dw>0
    nE, nr = Duw.shape
    logDw = np.log(blean*Dw+np.logical_not(blean)*epsilon)
    logDuw = np.log(Duw)
    dlnDwdr = d_dx(logDw, r)
    dlnDuwdr = d_dx(logDuw, r)
    r0 = r[n0]
    D0 = logDuw[:, n0-1]
    dDdr0 = dlnDuwdr[:, n0-1]
    r1 = r[n1]
    D1 = logDw[:, n1-1]
    dDdr1 = dlnDwdr[:, n1-1]
    #find interpolation that involves a constant second derivative of B from r0 to rbar                                                                                                                         
    #and a different constant second derivative of C from rbar to r1                                                                                                                                                                      
    BpC = 2*(dDdr1 - dDdr0)/(r1-r0)
    BmC = (8/(r1-r0)**2)*(D1-D0-(dDdr1+dDdr0)*(r1-r0)/2)
    B = (BpC+BmC)/2
    C = (BpC-BmC)/2
    rbar = (r0+r1)/2
    r_tr = r[1:n2+1]
    rr, rr0 = np.meshgrid(r_tr, r0*np.ones(nE))
    nrtr = len(r_tr)
    rr1 = r1*np.ones((nE, nrtr))
    D0D0 = np.repeat(D0.reshape(nE, 1), nrtr, axis=1)
    D1D1 = np.repeat(D1.reshape(nE, 1), nrtr, axis=1)
    Dpr0Dpr0 = np.repeat(dDdr0.reshape(nE, 1), nrtr, axis=1)
    Dpr1Dpr1 = np.repeat(dDdr1.reshape(nE, 1), nrtr, axis=1)
    rbarrbar = rbar*np.ones((nE, nrtr))
    BB = np.repeat(B.reshape(nE, 1), nrtr, axis=1)
    CC = np.repeat(C.reshape(nE, 1), nrtr, axis=1)
    ma0 = rr>rr0
    ma1 = rr<rr1
    #ma = np.full(logD.shape, False)                                                                                                                                                                                                        
    lessrbar = rr<rbarrbar
    Dnew_tr = ma0*ma1*(lessrbar*(D0D0 + Dpr0Dpr0*(rr-rr0)+0.5*BB*(rr-rr0)**2) + \
                     np.logical_not(lessrbar)*(D1D1 + Dpr1Dpr1*(rr-rr1)+0.5*CC*(rr-rr1)**2))
    Dnew_tr = Dnew_tr+np.logical_not(ma0)*logDuw + np.logical_not(ma1)*logDw
    return np.exp(Dnew_tr)

def smooth_diff(D_old, r, large=15):
    #rsmall=1000
    #nstart = np.where(np.abs(r-rsmall)==np.min(np.abs(r-rsmall)))[0][0]
    nstart = 1
    rstart = r[nstart]
    #this is where I will store the new diffusion coeficient (same dimensionality as the old)                                                                                                          
    Dbar_new = np.copy(D_old)
    #this is the copy of the diffusion coeffient that I will analyze (I am leaving off the first r value)                                                                                              
    Dbar = np.copy(Dbar_new[:, 1:])
    #compute log derivatives                                                                                                                                                                           
    blean = Dbar>0
    epsilon = 1e-300
    logDbar = np.log(blean*Dbar+np.logical_not(blean)*epsilon)
    print(logDbar.shape)
    dlnDdr = d_dx(logDbar, r)
    #plt.plot(r[nstart:-1], dlnDdr[100, nstart-1:-1])
    #plt.yscale('log')
    #check which energy values must must be smoothed 
    E_ind = np.unique(np.where(dlnDdr[:, nstart-1:-1]<0)[0])
    nE_trunc = len(E_ind)
    #lnD_trunc = logDbar[E_ind]                                                                                                                                                                        
    #dlnDdr = d_dx(lnD_trunc, r)                                                                                                                                                                       
    #d2lnDdr2 = d2_dx2(lnD_trunc, r)                                                                                                                                                                   
    d2lnDdr2 = d2_dx2(logDbar, r)
    L = np.abs(dlnDdr)/np.abs(np.logical_not(d2lnDdr2==0)*d2lnDdr2 + (d2lnDdr2==0)*epsilon)
    #find r1 and function value at r1 and derivative at r1                                                                                                                                             
    n1 = np.array([np.min(np.where(L[ind, nstart-1:-1]<large*(r[nstart:-1]-rstart))[0])+nstart for ind in E_ind])
    r1 = r[n1]
    D1 = logDbar[E_ind, n1-1]
    dDdr1 = dlnDdr[E_ind, n1-1]
    #find r2 and function value at r2 and derivative at r2                                                                                                                                             
    nend = -1
    rend=r[nend]
    n2 = np.array([np.max(np.where(L[ind, 1:nend]<large*(rend-r[2:nend]))[0]) + 2 for ind in E_ind])
    r2 = r[n2]
    D2 = logDbar[E_ind, n2-1]
    dDdr2 = dlnDdr[E_ind, n2-1]
    #find interpolation that involves a constant second derivative of B from x1                                                                                                                        
    #and a different constant second derivative of C from x_bar to x2                                                                                                                                  
    BpC = 2*(dDdr2 - dDdr1)/(r2-r1)
    BmC = (8/(r2-r1)**2)*(D2-D1-(dDdr2+dDdr1)*(r2-r1)/2)
    B = (BpC+BmC)/2
    C = (BpC-BmC)/2
    logD_new = np.copy(logDbar)
    rbar = (r1+r2)/2
    r_tr = r[1:]
    rr, rr1 = np.meshgrid(r_tr, r1)
    nrtr = len(r_tr)
    rr2 = np.repeat(r2.reshape(nE_trunc, 1), nrtr, axis=1)
    D1D1 = np.repeat(D1.reshape(nE_trunc, 1), nrtr, axis=1)
    D2D2 = np.repeat(D2.reshape(nE_trunc, 1), nrtr, axis=1)
    Dpr1Dpr1 = np.repeat(dDdr1.reshape(nE_trunc, 1), nrtr, axis=1)
    Dpr2Dpr2 = np.repeat(dDdr2.reshape(nE_trunc, 1), nrtr, axis=1)
    rbarrbar = np.repeat(rbar.reshape(nE_trunc, 1), nrtr, axis=1)
    BB = np.repeat(B.reshape(nE_trunc, 1), nrtr, axis=1)
    CC = np.repeat(C.reshape(nE_trunc, 1), nrtr, axis=1)
    ma_tr = np.logical_and(rr<rr2, rr>rr1)
    #ma = np.full(logD.shape, False)                                                                                                                                                                   
    lessrbar = rr<rbarrbar
    Dnew_tr = ma_tr*(lessrbar*(D1D1 + Dpr1Dpr1*(rr-rr1)+0.5*BB*(rr-rr1)**2) + \
        np.logical_not(lessrbar)*(D2D2 + Dpr2Dpr2*(rr-rr2)+0.5*CC*(rr-rr2)**2))
    Dnew_tr = Dnew_tr+np.logical_not(ma_tr)*logDbar[E_ind]
    logD_new[E_ind] = Dnew_tr
    #assume that the first r value is correct and replace the rest with the new values                                                                                                                 
    Dbar_new[:, 1:] = np.exp(logD_new)
    return Dbar_new

def smooth_diff_lowr(D, r, large=10):
    D_new = np.copy(D)
    epsilon=1e-300
    rsmall=1500
    rr, __ = np.meshgrid(r, D[:, 0])
    logD = np.log(D)
    dlnDdr = d_dx(logD, r)
    d2lnDdr2 = d2_dx2(logD, r)
    L = np.abs(dlnDdr)/np.abs(np.logical_not(d2lnDdr2==0)*d2lnDdr2 + (d2lnDdr2==0)*epsilon)
    nstart = np.where(np.abs(r-rsmall)==np.min(np.abs(r-rsmall)))[0][0]
    Eind = np.unique(np.where(L[:, 1:nstart]<large*(r[nstart]-rr[:, 1:nstart]))[0])
    n1 = np.array([np.max(np.where(L[ind, :nstart]<large*(r[nstart]-r[:nstart]))[0]) for ind in Eind])
    print('n1=', n1)
    nE_tr = len(Eind)
    r1 = r[n1]
    print('r1 = ', r1)
    D1 = logD[Eind, n1]
    print('D1 = ', D1)
    dDdr1 = dlnDdr[Eind, n1]
    print('dDdr1 = ', dDdr1)
    D0 = logD[Eind, 0]
    print('D0 = ', D0)
    B = 2*(D0-D1+r1*dDdr1)/r1**2
    rr, r1r1 = np.meshgrid(r, r1)
    nr = len(r)
    D1D1 = np.repeat(D1.reshape(nE_tr, 1), nr, axis=1)
    Dpr1Dpr1 = np.repeat(dDdr1.reshape(nE_tr, 1), nr, axis=1)
    BB = np.repeat(B.reshape(nE_tr, 1), nr, axis=1)
    ma_tr = rr<=r1r1
    logDnew_tr = ma_tr*(D1D1+(rr-r1r1)*Dpr1Dpr1+0.5*BB*(rr-r1r1)**2) + np.logical_not(ma_tr)*logD[Eind]
    D_new[Eind] = np.exp(logDnew_tr)
    return D_new

def d_dx(f, x, axis=-1, lin=True):
    num_axes = len(f.shape)
    if not axis==-1:
        ind = ['i', 'j', 'k', 'l', 'm', 'n']
        ind = ind[:num_axes]
        ind[axis], ind[-1] = ind[-1], ind[axis]
        s = ''.join(ind)
        f = np.einsum(s, f)
    if lin:
        dx = x[1]-x[0]
        sh = tuple(1 if i<num_axes-1 else 3 for i in range(num_axes))
        ddx1 = (1/(2*dx))*np.array([1, 0, -1])
        ddx = ddx1.reshape(sh)
        dfdx = convolve(f, ddx, mode='constant')
    else:
        pass
        #input matrix method for derivative                                                                                                                                                        
        #find dx vector                                                                                                                                                                            
# =============================================================================                                                                                                                    
#         dx_kernel = np.array([1, -1, 0])                                                                                                                                                         
#         dx = np.convolve(dx_kernel, x, 'same')                                                                                                                                                   
#         x3n = np.array([x[]])                                                                                                                                                                    
# =============================================================================                                                                                                                    
    if not axis==-1:
        dfdx = np.einsum(s, dfdx)
    return dfdx

#integrands                                                                                                                                                                                        
def binv_int(z, rval, E, D0):
    R = np.sqrt(rval**2-z**2)
    return 1/b(R, z, E)
def D1_int(z, rval, E, D0):
    R = np.sqrt(rval**2-z**2)
    return (D0/D0_bench)*D(R, z, E)/b(R, z, E)
def D2_int(z, rval, E, D0):
    R = np.sqrt(rval**2-z**2)
    dDdr = -(D0/D0_bench)*(D(R, z, E)/(3*B(R, z)))*(dBdR(R, z)*R/rval+dBdz(R, z)*z/rval)
    return dDdr/b(R, z, E)
def b_int(z, rval, E, D0):
    R = np.sqrt(rval**2-z**2)
    return b(R, z, E)
def D_int(z, rval, E, D0):
    R = np.sqrt(rval**2-z**2)
    return (D0/D0_bench)* D(R, z, E)

#updated integrands                                                                                                                                                                                
def tau_int(z, rval, E, D0):
    R = np.sqrt(rval**2-z**2)
    res = np.logical_not(rval==0)*(1/(b(R,z,E)/E + (D0/D0_bench)*D(R,z,E)/((cf.cm_per_pc*rval + 0.000001)**2)))
    return res
def D_tau1_int(z, rval, E, D0):
    R = np.sqrt(rval**2-z**2)
    res = np.logical_not(rval==0)*(D0/D0_bench)*D(R,z,E)/(b(R,z,E)/E + (D0/D0_bench)*D(R,z,E)/((cf.cm_per_pc*rval+0.000001)**2))
    return res
def D_tau2_int(z, rval, E, D0):
    R = np.sqrt(rval**2-z**2)
    dDdr = -(D(R, z, E)/(3*B(R, z)))*(dBdR(R, z)*R/rval+dBdz(R, z)*z/rval)
    return np.logical_not(rval==0)*(D0/D0_bench)*dDdr/(b(R,z,E)/E + (D0/D0_bench)*D(R,z,E)/((cf.cm_per_pc*rval+0.000001)**2))
def b_tau_int(z, rval, E, D0):
    R = np.sqrt(rval**2-z**2)
    return np.logical_not(rval==0)*(b(R, z, E)/(b(R,z,E)/E + (D0/D0_bench)*D(R,z,E)/((cf.cm_per_pc*rval+0.000001)**2)))

#find model for diffusion coeff                                                                                                                                                                   
#r should have legnth nr                                                                                                                                                                          
def find_D_model(r, Dbar):
    nE, nr = Dbar.shape
    scores = np.zeros(nE)
    coefs = np.zeros(nE)
    ints = np.zeros(nE)
    for j in range(nE):
        inds = np.where(Dbar[j]>0)[0]
        r_tr = np.array([r[inds]]).T
        D_tr = Dbar[j][inds]
        lnD = np.log(np.array([D_tr])).T
        reg = LinearRegression().fit(r_tr, lnD)
        scores[j] = reg.score(r_tr, lnD)
        coefs[j] = reg.coef_[0,0]
        ints[j] = reg.intercept_[0]
    rr, ii = np.meshgrid(r, ints)
    _, cc = np.meshgrid(r, coefs)
    Dbar_model = np.exp(ii)*np.exp(cc*rr)
    return Dbar_model, np.min(scores)

