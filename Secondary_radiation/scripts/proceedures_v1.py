#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:12:11 2021

@author: mitchellweikert
"""
import numpy as np
import constants_and_functions as cf
from scipy.interpolate import interp1d, interp2d, UnivariateSpline
from scipy.integrate import quad
from scipy.special import zeta, factorial
import meta_variables as mv
from scipy.sparse import diags
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import vegas
import time
import os.path

path_name = os.path.realpath(__file__)
path_base = path_name.split('scripts')[0]
def gen_radial_funcs(B_model, rho_star_model, DM_model):
    if not (B_model == None):
        def B(r):
            return cf.dist(r, B_model)
    else:
        B = None
        
    if not (B_model == None):  
        def dBdr(r):
            return cf.ddistdr(r, B_model)
    else:
        dBdr = None
        
    if not (rho_star_model == None):   
        def rho_star(r): 
            return cf.dist(r, rho_star_model)
    else:
        rho_star = None
        
    if not (DM_model == None):  
        def rho_DM(r):
            return cf.dist(r, DM_model)
    else:
        rho_DM = None
        
    return B, dBdr, rho_star, rho_DM

def gen_funcs_sync_ic(fe_arrays, f_star_params, rho_star):
    rr = fe_arrays[0]
    EE = fe_arrays[1]
    u = fe_arrays[2]
    z_ind = np.where(u==0)
    if len(z_ind[0])>0:
        min_zind = np.min(z_ind[0])
        logfe1 = np.log(u[:min_zind]/rr[:min_zind])
        logfe_int1 = interp2d(rr[0], EE[:min_zind,0], logfe1, kind='cubic')
        for i in range(len(z_ind[0])):
            u[z_ind[0][i]][z_ind[1][i]] = 1e-300
        logfe2 = np.log(u[min_zind-1:]/rr[min_zind-1:])
        logfe_int2 = interp2d(rr[0], EE[min_zind-1:, 0], logfe2, kind='linear')
    else:
        logfe1 = np.log(u/rr)
        logfe_int1 = interp2d(rr[0], EE[:,0], logfe1, kind='cubic')
        logfe_int2 = None
        min_zind = EE.shape[0]-1

    def fe(r, E, Ecut=EE[min_zind][0]):
        if logfe_int2==None:
            return np.exp(logfe_int1(r, E))[0]
        else:
            return np.piecewise(E, [E<Ecut, E>=Ecut], [lambda E: np.exp(logfe_int1(r, E))[0], lambda E: np.exp(logfe_int2(r, E))[0]]) 
    
    if not (f_star_params.any()==None or rho_star == None):
        T0 = f_star_params[0]
        delta_T = f_star_params[1]
        E_star_max = 5*(T0+5*delta_T)
        E_star = np.linspace(0, E_star_max, 400)
        integral = np.zeros(len(E_star))
        for i in range(len(E_star)-1):
            integral[i+1] = quad(cf.T_dist, 0, np.inf, args=(E_star[i+1],[T0, delta_T]))[0]
    
        spectrum = interp1d(E_star, integral, kind='cubic')
        norm = factorial(3)*zeta(4)*T0**4/(np.pi**2*(cf.h_bar*cf.c)**3)*(1+6*(delta_T/T0)**2 + 3*(delta_T/T0)**4) #eV/cm^3
        
        def f_star(E, r):
            E_bool = E<E_star_max
            E_bool = E_bool.astype(int)
            f = ((rho_star(r)/norm)/(np.pi**2*np.sqrt(2*np.pi)*delta_T))/(cf.h_bar*cf.c)**3*spectrum(E*E_bool) #cm^-3 ev^-1
            return f
    else:
        f_star = None
    
    #plt.plot(np.linspace(0, E_star_max, 100), f_star(np.linspace(0, E_star_max, 100), 1000))
    #plt.show()
    return fe, f_star

#f must be UnivariateSpline object
def lin_extrap(f, xmin, xmax):
    def fex(u):
        res = np.piecewise(u, [u<xmin or u>xmax], [np.log(1e-300), lambda u: f(u)]) 
        return res
    return fex
            
def find_equillibrium(mx, sigma_v, D0_cm, R, e_spec, e_bins, funcs=None, B_model=None, rho_star_model=None, DM_model=None, E_spacing='log'):
    sol = []
    rr, EE, dr_v, dE_v = mv.grid(mv.nr, mv.nE, [0, R], [cf.me, mx], E_spacing=E_spacing)
    dr = dr_v[0]
    if not (B_model==None and rho_star_model==None and DM_model==None):
        B_r, dBdr_r, rho_star_r, rho_DM_r = gen_radial_funcs(B_model, rho_star_model, DM_model)
        r0_DM = DM_model[2]
    else:
        B_r = funcs[0]
        dBdr_r = funcs[1]
        rho_star_r = funcs[2]
        rho_DM_r = funcs[3]
    
    #interpolate and exterpolate spectra
    ebins_cent = np.zeros(len(e_bins)-1)
    for i in range(len(e_bins)-1):
        ebins_cent[i] = (e_bins[i]+e_bins[i+1])/2
    ebins_min = ebins_cent[0]
    ebins_max = ebins_cent[-1]
    ebins_cent = np.insert(ebins_cent, 0, cf.me)
    e_spec = np.insert(e_spec, 0, 0)
    ebins_cent = np.append(ebins_cent, mx)
    e_spec = np.append(e_spec, 0)
    
    dNdE_int = UnivariateSpline(ebins_cent, e_spec, k=3, s=0)
    E_v = EE[:, 0]
    dNdE_v = dNdE_int(E_v)
    neg_ind = np.where(dNdE_v<0)[0]
    for i in range(len(neg_ind)):
        dNdE_v[neg_ind[i]] = 0
    
# =============================================================================
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.plot(E_v, dNdE_v)
#     ax.set_xscale('log')
#     plt.show()
# =============================================================================
    
    
    #magnetic field and radiation field
    B = B_r(rr)
    dBdr = dBdr_r(rr)
    rho_star = rho_star_r(rr)
    
    #dark matter distribution
    rho_DM = rho_DM_r(rr)
    
    #diffusion and loss (powers of 1/3 and 2 are hard-coded in)
    D0 = D0_cm/cf.cm_per_pc**2
    D = (D0/B**(1/3))*(EE**(1/3))
    dDdr = -(1/3)*dBdr*D/B
    b = (cf.b0_IC*(1+rho_star/cf.rho_CMB)+ cf.b0_syn*B**2)*(EE**2)
    dbdE = 2*(cf.b0_IC*(1+rho_star/cf.rho_CMB) + cf.b0_syn*B**2)*EE
    
    #source function
    rr, dNdE = np.meshgrid(rr[0], dNdE_v)
    Q_cm = (sigma_v/(2*mx**2))*rho_DM**2*dNdE #1/(s*cm^3*GeV)
    Q = (cf.cm_per_pc**3)*Q_cm #1/(s*pc^3*GeV)
    
    #extract DM scale radius (assume nfw-like profile if DM-model is None)
    if not (DM_model == None):
        r0_DM = DM_model[2]
    else:
        r_small = 1 #pc
        delta_r = .1 #pc
        r_large = 20000 #pc
        gamma = -r_small*(np.log(rho_DM_r(r_small + delta_r))-np.log(rho_DM_r(r_small)))/delta_r
        r0_DM = -(3-gamma)*(((np.log(rho_DM_r(r_large + delta_r))-np.log(rho_DM_r(r_large)))/delta_r+gamma/r_large)**(-1))-r_large
        print('DM scale radius approximately: ', r0_DM)
    #start dt_0 at longest timescale
    maxtau_D = r0_DM**2/np.amin(D)
    maxtau_b = (mx-cf.me)/np.amin(b)
    tau_ch_ep = np.sqrt((r0_DM**2/D)**2 + ((mx-cf.me)/b)**2)
    max_tau = max(maxtau_D, maxtau_b)
    dt_0 = max_tau
    
    dr_mat, dE_mat = np.meshgrid(dr_v, dE_v)
    #initial finite difference coefficients for:
    #r dereivatives
    a3_0 = (D/(dr**2) - dDdr/(2*dr))*dt_0
    a2_0 = (-2*D/(dr**2)- dDdr/rr)*dt_0
    a1_0 = (D/(dr**2) + dDdr/(2*dr))*dt_0
    A_0 = np.zeros((mv.nE, mv.nr-1, mv.nr-1))
    for j in range(0,mv.nE):
        A_0[j] = diags([a3_0[j][1:], a2_0[j], a1_0[j][0:-1]], [-1, 0, 1], shape=(mv.nr-1, mv.nr-1)).toarray() 
    #E derivatives   
    b3_0 = 0*np.transpose(b)
    b2_0 = -(np.transpose(b)/np.transpose(dE_mat)[1:])*dt_0 
    b1_0 = (np.transpose(b)[:,1:]/np.transpose(dE_mat)[1:,0:-1])*dt_0
    B_0 = np.zeros((mv.nr-1, mv.nE, mv.nE))
    for i in range(0,mv.nr-1):
        B_0[i] = diags([b3_0[i][1:], b2_0[i], b1_0[i]], [-1, 0, 1], shape=(mv.nE, mv.nE)).toarray()
    
    #shortest timescales
    max_a1 = np.amax(a1_0)
    max_a2 = np.amax(a2_0)
    max_a3 = np.amax(a3_0)
    max_b1 = np.amax(b1_0)
    max_b2 = np.amax(b2_0)
    max_b3 = np.amax(b3_0)
    max_coeff = max(max_a1, max_a2, max_a3, max_b1, max_b2, max_b3)
    tau_min = dt_0/max_coeff
    
    #initialize for main loop
    u = 0*D
    dt = dt_0
    reduce = True
    
    #this is the main loop. It iteratively reduces dt by a factor of 2 starting from dt=dt_0 until either dt becomes significantly smaller than the smallest time-scale in the problem or
    #the solution from the last three values of dt are very similar. The solution from one value of dt is fed in as an initial condition for the next value of dt. After each value of dt
    #fe = log(u/r) vs. r is plotted in 3D for the solution, u, and the solution is added to the variable sol
    while reduce:
       #rescale matrices for this time step
        a1 = a1_0*dt/dt_0
        a2 = a2_0*dt/dt_0
        a3 = a3_0*dt/dt_0
        A = A_0*dt/dt_0
        b1 = b1_0*dt/dt_0
        b2 = b2_0*dt/dt_0
        b3 = b3_0*dt/dt_0
        B = B_0*dt/dt_0
        
        #reformat for tridiag_plus
        a_v = -a3.reshape((mv.nr-1)*mv.nE)
        c_v = -a1.reshape((mv.nr-1)*mv.nE)
        for i in range(0,mv.nE):
            a_v[(mv.nr-1)*i] = 0
            c_v[(mv.nr-1)*(i+1)-1] = 0
        b_v = (1-a2-np.transpose(b2)).reshape((mv.nr-1)*mv.nE)
        e_v = -np.transpose(b1).reshape((mv.nr-1)*(mv.nE-1))
        dQ_v = (rr*Q*dt).reshape((mv.nr-1)*mv.nE)
        
        #initial conditions of u
        t_step = 0
        iterate = True
        if dt==dt_0:
            u = np.zeros((mv.nt+1, mv.nE, mv.nr-1))
        else:
            u0 = u[-1]
            u = np.zeros((mv.nt+1, mv.nE, mv.nr-1))   
            u[0] = u0
        
        # this loop steps through time and updates u based off of backward differences using sparse matrix scheme tridiag_plus (from constants_and_functions module). When u stops changing
        # significantly, the loop exits and the final value of u is used as the solution from the present time-step. The initial value of u for the present time-step is set to 0 when 
        #dt=dt_0 and it is set to the solution from the previous time-step when dt<dt_0
        while iterate:
            if t_step>=mv.nt:
                u = np.concatenate((u, np.zeros((1, mv.nE, mv.nr-1))), axis=0)
            du_cur_v = u[t_step].reshape((mv.nr-1)*mv.nE)
            d_v = dQ_v+du_cur_v
            u_next_v = cf.tridiag_plus(a_v, b_v, c_v, d_v, e_v)
            
            #update u
            u[t_step+1] = u_next_v.reshape((mv.nE, mv.nr-1))
            
            #check for convergence and advance t_step
            done = False
            for j in range(0, mv.nr-1):
                for k in range(0, mv.nE):
                    if u[t_step][k][j] == 0 and u[t_step+1][k][j] != 0:
                        perc_ch = 1
                    elif u[t_step][k][j] != 0: 
                        perc_ch = np.absolute(u[t_step+1][k][j] - u[t_step][k][j])/u[t_step][k][j]
                    else:
                        perc_ch = 0
                    if perc_ch > 0.001:
                        t_step = t_step+1
                        done = True
                        break
                    elif j == mv.nr-2 and k == mv.nE-1:
                        iterate = False
                        dt = dt/2
                if done:
                    break
        
        #truncate u if needed
        if t_step < mv.nt-1:
            u = u[0:t_step+2]
        #store u in sol
        sol.append(u)
        print('dt = ' +  '{:.2e}'.format(dt))
    
        #fig = plt.figure()
        #ax = Axes3D(fig)
        #ax.plot_surface(rr, EE, np.log10(u[-1]/rr))
        #ax.set_xlabel('r (pc)', fontsize=18)
        #ax.set_ylabel('E (GeV)', fontsize = 18)
        #ax.set_zlabel('log[f/((GeV*pc^3)^(-1))]', fontsize=18)
        #ax.set_title('dt = ' + "{:.2e}".format(dt) + ' s', fontsize=20)
        #ax.view_init(30,30)
        #plt.show()
        if dt < tau_min/100:
            reduce = False
        elif len(sol)>=5:
            if sol[-1].shape[0] == sol[-2].shape[0] == sol[-3].shape[0] == 2:
                reduce = False
    
    return rr, EE, u[-1]

def compute_sync(nu, rho, mx, R, D, funcs=None, B_model=None, rho_star_model=None, fe_arrays=None, f_star_params=None):
    
    if not (B_model == None and rho_star_model==None):
        B_r, dBdr_r, rho_star_r, rho_DM = gen_radial_funcs(B_model, rho_star_model, None)
    else:
        B_r = funcs[0]
        
    if not (fe_arrays == None and f_star_params == None):
        fe, f_star = gen_funcs_sync_ic(fe_arrays, f_star_params, rho_star_r)
    else:
        fe = funcs[1]
        
    def sync_integrand(integ_vars):
        return cf.sync_integrand(integ_vars, [nu, rho], [B_r, fe])
    
    l_lim = np.sqrt((R*mv.rr_unit[-1][-1])**2-rho**2)
    #plot integrand
# =============================================================================
#     y_v = np.linspace(0.1, .9, 3)
#     th_v = np.linspace(.1, np.pi-.1, 3)
#     gamma_v = np.linspace(2, mx/(2*cf.me), 3)
#     l_v = np.linspace(0, l_lim-1000, 3)
#     #iterate over interation vars
#     fig  = plt.figure()
#     for n in range(4):
#         print('plotting figs '+ str(n))
#         if n==0:
#             for i in range(len(th_v)):
#                 for j in range(len(gamma_v)):
#                     for k in range(len(l_v)):
#                         y_fine = np.logspace(-3, 0, 1000)
#                         f_fine = sync_integrand([y_fine, th_v[i], gamma_v[j], l_v[k]])
#                         ax = fig.add_subplot()
#                         ax.plot(y_fine, f_fine)
#                         plt.title('th = '+ str(np.round(th_v[i], 2)) + ' gamma = ' + str(np.round(gamma_v[j], 2)) + ' l = '+ str(np.round(l_v[k])))
#                         plt.xlabel('y')
#                         plt.ylabel('f')
#                         plt.savefig(path_base + '/figs/' + 'y_indep' + '_th_'+ str(np.round(th_v[i], 2)) + '_gamma_' + str(np.round(gamma_v[j], 2)) + '_l_'+ str(np.round(l_v[k]))+'.pdf')
#                         plt.clf()
#         elif n==1:
#             for i in range(len(th_v)):
#                 for j in range(len(gamma_v)):
#                     for k in range(len(l_v)):
#                         th_fine = np.linspace(.001, np.pi-.001, 1000)
#                         f_fine = sync_integrand([y_v[i], th_fine, gamma_v[j], l_v[k]])
#                         ax = fig.add_subplot()
#                         ax.plot(th_fine, f_fine)
#                         plt.title('y = '+ str(np.round(y_v[i], 2)) + ' gamma = ' + str(np.round(gamma_v[j], 2)) + ' l = '+ str(np.round(l_v[k])))
#                         plt.xlabel('th (rad)')
#                         plt.ylabel('f')
#                         plt.savefig(path_base + '/figs/' + 'th_indep' + '_y_'+ str(np.round(y_v[i], 2)) + '_gamma_' + str(np.round(gamma_v[j], 2)) + '_l_'+ str(np.round(l_v[k]))+'.pdf')
#                         plt.clf()
#         elif n==2:
#             for i in range(len(y_v)):
#                 for j in range(len(th_v)):
#                     for k in range(len(l_v)):
#                         gamma_fine = np.logspace(0, np.log10(mx/cf.me), 10000)
#                         f_fine = sync_integrand([y_v[i], th_v[j], gamma_fine, l_v[k]])
#                         ax = fig.add_subplot()
#                         ax.plot(gamma_fine, f_fine)
#                         plt.title('y = '+ str(np.round(y_v[i], 2)) + ' th = ' + str(np.round(th_v[j], 2)) + ' l = '+ str(np.round(l_v[k])))
#                         plt.xlabel('gamma')
#                         plt.ylabel('f')
#                         plt.savefig(path_base + '/figs/' + 'gamma_indep' + '_y_'+ str(np.round(y_v[i], 2)) + '_th_' + str(np.round(th_v[j], 2)) + '_l_'+ str(np.round(l_v[k])) + '.pdf')
#                         plt.clf()
#         elif n==3:
#             for i in range(len(y_v)):
#                 for j in range(len(th_v)):
#                     for k in range(len(gamma_v)):
#                         l_fine = np.linspace(-l_lim, l_lim, 1000)
#                         f_fine = sync_integrand([y_v[i], th_v[j], gamma_v[k], l_fine])
#                         ax = fig.add_subplot()
#                         ax.plot(l_fine, f_fine)
#                         plt.title('y = '+ str(np.round(y_v[i], 2)) + ' th = ' + str(np.round(th_v[j], 2)) + ' gamma = '+ str(np.round(gamma_v[k])))
#                         plt.xlabel('l (pc)')
#                         plt.ylabel('f')
#                         plt.savefig(path_base + '/figs/' + 'l_indep' + '_y_'+ str(np.round(y_v[i], 2)) + '_th_' + str(np.round(th_v[j], 2)) + '_gamma_'+ str(np.round(gamma_v[k])) + '.pdf')
#                         plt.clf()
# =============================================================================
                        
    integ = vegas.Integrator([[0, 1], [0, np.pi], [1, mx/cf.me], [-l_lim, l_lim]])
    result = integ(sync_integrand, nitn=7, neval=60000)
    nu_dSdnudth = (nu/((cf.cm_per_pc)**2))*np.array([result.mean, result.sdev])*2*np.pi*rho/np.sqrt(rho**2+D**2)
    
    return nu_dSdnudth
    
def compute_ic(nu, rho, mx, R, D, funcs=None, B_model=None, rho_star_model=None, fe_arrays=None, f_star_params=None):
    
    if not (B_model == None and rho_star_model==None):
        B, dBdr, rho_star, rho_DM = gen_radial_funcs(B_model, rho_star_model, None)
    else:
        rho_star = funcs[0]
        
    if not (fe_arrays == None and f_star_params == None):
        fe, f_star = gen_funcs_sync_ic(fe_arrays, f_star_params, rho_star)
    else:
        fe = funcs[1]
        f_star = funcs[2]
        
    def ic_integrand(integ_vars):
        return cf.ic_integrand(integ_vars, [nu,rho], [fe, f_star, cf.f_CMB])
    
    l_lim = np.sqrt((R*mv.rr_unit[-1][-1])**2-rho**2)
    integ = vegas.Integrator([[0, 1], [1, mx/cf.me], [-l_lim, l_lim]])
    result = integ(ic_integrand, nitn=7, neval=60000)
    nudSdnudth = (cf.erg_per_ev*((2*np.pi*nu*cf.h_bar)**2)/(cf.cm_per_pc**2))*np.array([result.mean, result.sdev])*2*np.pi*rho/np.sqrt(rho**2+D**2)
    
    return nudSdnudth

#test proceedures
if __name__ == '__main__':
    path_name = '/Users/mitchellweikert/Documents/Rutgers/Multiwavelegnth_project/'
    output_type = ['electron_spectra', 'phasespace_density_times_r', 'sync_nudSdnudth_vs_nu_rho', 'ic_nudSdnudth_vs_nu_rho']
    B_model = [['exp', 10, 1500], ['exp', 7, 20000]]
    rho_star_model = [['exp', 8, 4300]]
    DM_model = [['nfw', 0.418, 1.25, 16500]]
    start = time.time()
    B_r, dBdr_r, rho_star_r, rho_DM_r = gen_radial_funcs(B_model, rho_star_model, DM_model)
    rad_func_elapsed = time.time() - start
    print('generate radial function elapsed time: ', rad_func_elapsed, ' s')
    mx = 50
    Rmax = 50000
    DA = 780000
    channel = 'bb_bar'
    spec_vars_str = '___mx__' + str(mx) + '___channel__' + channel
    spec_file_str = path_name + '/' + output_type[0] + '/' + output_type[0] + spec_vars_str + '.npy'
    dNdE_v = np.load(spec_file_str)
    start = time.time()
    rr, EE, u = find_equillibrium(mx, 2.2e-26, 3e28, Rmax, dNdE_v, funcs=[B_r, dBdr_r, rho_star_r, rho_DM_r], B_model=None, rho_star_model=None, DM_model=None)
    equil_elapsed = time.time()-start
    print('equillibrium distribution elapsed time: ', equil_elapsed, ' s')
    f_star_params = [cf.kb*5500, cf.kb*1000]
    def B_r_G(r):
        return (1e-6)*B_r(r)
    start = time.time()
    fe, f_star = gen_funcs_sync_ic([rr, EE, u], f_star_params, rho_star_r)
    ic_sync_funcs_time = time.time() - start
    print('generate ic and sync functions elapsed time: ', ic_sync_funcs_time, ' s')
    nu_sync = 1e10 #Hz
    rho_sync = 1000 #pc
    E_ic = 1000 #eV
    nu_ic = E_ic/(2*np.pi*cf.h_bar) #Hz
    rho_ic = 1000 #pc
    start = time.time()
    nu_dSdnudth_sync = compute_sync(nu_sync, rho_sync, mx, Rmax, DA, funcs=[B_r_G, fe])
    compute_sync_time = time.time()-start
    print('compute sync elapsed time: ', compute_sync_time, ' s')
    start= time.time()
    nu_dSdndth_ic = compute_ic(nu_ic, rho_ic, mx, Rmax, DA, funcs=[rho_star_r, fe, f_star])
    compute_ic_time = time.time()-start
    print('compute ic elapsed time: ', compute_ic_time, ' s')
    
