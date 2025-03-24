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

def smooth_fe(fe_arrays):
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
    
    return fe
    
    
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
            
def find_equillibrium(mx, sigma_v, D0_cm, R, e_spec, e_bins, am, spherical_ave='unweighted', DM_model=None, D0_bench=3e28, E_spacing='log'):
    sol = []
    #make grid with 3 extra r values, one extra at r=0 and two extra at r=R and r=R-dr
    rr, EE, dr_v, dE_v = mv.grid(mv.nr+3, mv.nE+2, [0, R], [cf.me, mx], E_spacing=E_spacing)
    dr = dr_v[0]
    #determine spherically averaged diffusion and loss coefficients
    r = rr[0]
    E = EE[:, 0]
    if spherical_ave=='weighted':
        sph_avg_terms = am.spherical_average_terms([am.binv_int, am.D1_int, am.D2_int], r, E, D0_cm)
        b, Dinit = am.find_coefficients(sph_avg_terms, r, E, DM_model)
        Dsmooth = am.smooth_diff(Dinit, r[:-1])
        D = am.smooth_diff_lowr(Dsmooth, r[:-1])
    elif spherical_ave=='weighted_dl':
        sph_avg_terms = am.spherical_average_terms([am.tau_int, am.D_tau1_int, am.D_tau2_int, am.b_tau_int, am.D_int], r, E, D0_cm)
        b, D_w = am.find_coefficients(sph_avg_terms, r, E, DM_model)
        D_uw = sph_avg_terms[-1]
        r0 = 100
        #replace solution at small r with \langle D \rangle
        Dnew = am.merge_D(D_w, D_uw, r, r0)
        n2 = Dnew.shape[1]
        D_w[:, 1:n2+1] = Dnew
        D = am.smooth_diff(D_w, r[:-1])
    elif spherical_ave=='unweighted':
        b, D = am.spherical_average_terms([am.b_int, am.D_int], r[:-1], E, D0_cm)

    dDdr = am.d_dx(D, r[:-1])
    b = b[1:-1, 1:-1]
    D = D[1:-1, 1:-1]
    dDdr = dDdr[1:-1, 1:-1]
    print('zero inds: ', np.where(D==0))
    #interpolate and exterpolate spectra
    ebins_cent = np.array([(e_bins[i]+e_bins[i+1])/2 for i in range(len(e_bins)-1)])
    ebins_min = ebins_cent[0]
    ebins_max = ebins_cent[-1]
    ebins_cent = np.insert(ebins_cent, 0, cf.me)
    e_spec = np.insert(e_spec, 0, 0)
    ebins_cent = np.append(ebins_cent, mx)
    e_spec = np.append(e_spec, 0)
    dNdE_int = UnivariateSpline(ebins_cent, e_spec, k=3, s=0)

    #redefine E and r grids
    r = r[1:-2]
    E = E[1:-1]
    rr, EE = np.meshgrid(r, E)

    dNdE_v = dNdE_int(E)
    neg_ind = np.where(dNdE_v<0)[0]
    dNdE_v[neg_ind] = 0
    
# =============================================================================
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.plot(E_v, dNdE_v)
#     ax.set_xscale('log')
#     plt.show()
# =============================================================================
    
    #dark matter distribution
    rho_DM = cf.dist(rr, DM_model)
    
    #scale and change units for diffusion coefficient
    D0_pc = D0_cm/cf.cm_per_pc**2
    D = (D0_pc/D0_bench)*D
    dDdr = (D0_pc/D0_bench)*dDdr
    
    #source function
    _, dNdE = np.meshgrid(r, dNdE_v)
    Q_cm = (sigma_v/(2*mx**2))*rho_DM**2*dNdE #1/(s*cm^3*GeV)
    Q = (cf.cm_per_pc**3)*Q_cm #1/(s*pc^3*GeV)
    #extract DM scale radius
    r0_DM = DM_model[0][3]
    
    #start dt_0 at longest timescale
    maxtau_D = r0_DM**2/np.amin(D)
    print('maxtaud_D = ', maxtau_D)
    print('index acheived', np.where(D==np.amin(D)))
    maxtau_b = (mx-cf.me)/np.amin(b)
    print('maxtau_b = ', maxtau_b)
    print('index acheived', np.where(b==np.amin(b)))
    tau_ch_ep = np.sqrt((r0_DM**2/D)**2 + ((mx-cf.me)/b)**2)
    max_tau = max(maxtau_D, maxtau_b)
    dt_0 = max_tau
    print('dt_0 = ', dt_0)
    
    _, dE_mat = np.meshgrid(r, dE_v[1:])
    #initial finite difference coefficients for:
    #r dereivatives
    
    a3_0 = (D/(dr**2) - dDdr/(2*dr))*dt_0
    a2_0 = (-2*D/(dr**2)- dDdr/r)*dt_0
    a1_0 = (D/(dr**2) + dDdr/(2*dr))*dt_0
    #A_0 = np.array([diags([a3_0[j][1:], a2_0[j], a1_0[j][0:-1]], [-1, 0, 1], shape=(mv.nr, mv.nr)).toarray() for j in range(mv.nE)])
    #E derivatives   
    b3_0 = 0*np.transpose(b)
    b2_0 = -(np.transpose(b)/np.transpose(dE_mat))*dt_0 
    b1_0 = (np.transpose(b)[:,1:]/np.transpose(dE_mat)[:, :-1])*dt_0
    #B_0 = np.array([diags([b3_0[i, 1:], b2_0[i], b1_0[i]], [-1, 0, 1], shape=(mv.nE, mv.nE)).toarray() for i in range(mv.nr)])
    
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
        #A = A_0*dt/dt_0
        b1 = b1_0*dt/dt_0
        b2 = b2_0*dt/dt_0
        b3 = b3_0*dt/dt_0
        #B = B_0*dt/dt_0
        
        #reformat for tridiag_plus
        a_v = -a3.reshape((mv.nr)*mv.nE)
        c_v = -a1.reshape((mv.nr)*mv.nE)
        for i in range(0,mv.nE):
            a_v[(mv.nr)*i] = 0
            c_v[(mv.nr)*(i+1)-1] = 0
        b_v = (1-a2-np.transpose(b2)).reshape((mv.nr)*mv.nE)
        e_v = -np.transpose(b1).reshape((mv.nr)*(mv.nE-1))
        dQ_v = (rr*Q*dt).reshape((mv.nr)*mv.nE)
        
        #initial conditions of u
        t_step = 0
        iterate = True
        if dt==dt_0:
            u = np.zeros((mv.nt+1, mv.nE, mv.nr))
        else:
            u0 = u[-1]
            u = np.zeros((mv.nt+1, mv.nE, mv.nr))   
            u[0] = u0
        
        # this loop steps through time and updates u based off of backward differences using sparse matrix scheme tridiag_plus (from constants_and_functions module). When u stops changing
        # significantly, the loop exits and the final value of u is used as the solution from the present time-step. The initial value of u for the present time-step is set to 0 when 
        #dt=dt_0 and it is set to the solution from the previous time-step when dt<dt_0
        while iterate:
            if t_step>=mv.nt:
                u = np.concatenate((u, np.zeros((1, mv.nE, mv.nr))), axis=0)
            du_cur_v = u[t_step].reshape((mv.nr)*mv.nE)
            d_v = dQ_v+du_cur_v
            u_next_v = cf.tridiag_plus(a_v, b_v, c_v, d_v, e_v)
            
            #update u
            u[t_step+1] = u_next_v.reshape((mv.nE, mv.nr))
            
            #check for convergence and advance t_step
            done = False
            for j in range(0, mv.nr):
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
                    elif j == mv.nr-1 and k == mv.nE-1:
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

def compute_sync(nu, rho, phi, mx, rmax, DA, beta, B=None, fe_func=None, fe_arrays=None):
    
    print('compute sync running for nu = ' + '{:.2e}'.format(nu) + ' Hz; ' + \
          'rho = ' + str(np.round(rho, 1)) + ' pc; ' + 'phi = ' + str(np.round(phi, 2)) + 'rad')
    nu_max = 10*(cf.e*B(0,0)/(cf.c*cf.me*cf.g_per_GeV))*(mx/cf.me)**2/(2*np.pi)
    
    if not (fe_func == None):
        fe = fe_func
    elif not (fe_arrays == None):
        fe = smooth_fe(fe_arrays)
        rr = fe_arrays[0]
    
    def sync_integrand(integ_vars):
        return cf.sync_integrand(integ_vars, [nu, rho, phi], [B, fe], [beta])
    if nu<nu_max:
        l_lim = np.sqrt(rmax**2-rho**2)
        integ = vegas.Integrator([[0, 1], [0, np.pi], [1, mx/cf.me], [-l_lim, l_lim]])
        result = integ(sync_integrand, nitn=7, neval=60000)
        nu_dSdnudth = (nu/((cf.cm_per_pc)**2))*np.array([result.mean, result.sdev])
    else:
        print('nu value requested is larger than the max nu value at which synchrotron \
              can be produced for this situation. Returning 0')
        nu_dSnudth = np.array([0,0])
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
    am = __import__('Andromeda1')
    spherical_ave = 'weighted'
    mx=24.3
    D0_cm = 3e27
    D0_bench = 3e28
    sigmav = 2.2e-26
    rmax=50000
    DM_model=[['nfw', 0.43, 1.25, 16500.0]]
    E_spacing='log'
    sol = []
    #make grid with 3 extra r values, one extra at r=0 and two extra at r=R and r=R-dr
    rr, EE, dr_v, dE_v = mv.grid(mv.nr+3, mv.nE+2, [0, rmax], [cf.me, mx], E_spacing=E_spacing)
    dr = dr_v[0]
    #determine spherically averaged diffusion and loss coefficients
    r = rr[0]
    E = EE[:, 0]
    if spherical_ave=='weighted':
        sph_avg_terms = am.spherical_average_terms([am.binv_int, am.D1_int, am.D2_int], r, E)
        b, Dinit = am.find_coefficients(sph_avg_terms, r, E, DM_model)
        Dsmooth = am.smooth_diff(Dinit, r[:-1])
        Dsmooth2 = am.smooth_diff_lowr(Dsmooth, r[:-1])
    elif spherical_ave=='unweighted':
        b, D = am.spherical_average_terms([am.b_int, am.D_int], r, E)

    dDdr = am.d_dx(Dsmooth2, r[:-1])
    b = b[1:-1, 1:-1]
    D = Dsmooth2[1:-1, 1:-1]
    dDdr = dDdr[1:-1, 1:-1]
    
    
