#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:44:25 2021

@author: mitchellweikert
"""
import numpy as np
from scipy.sparse import diags
from scipy.special import kv

#define fundamental constants
b0_IC = 2.69*10**(-17)  #GeV/s
b0_syn = 2.52*10**(-18)  #GeV/s
me = 5.10999*10**(-4)  #GeV
e = 4.803*10**(-10) #statcoulombs
c = 2.998*10**10 #cm/s
h_bar = 6.582e-16 #eV*s
T_CMB = 2.348e-4 #eV
rho_CMB = np.pi**2*T_CMB**4/(15*(c*h_bar)**3) #eV/cm^3
erg_per_ev = 1.602e-12
sigma_T = 6.652e-25 #cm^2
g_per_GeV = 1.783e-24 #g/GeV
cm_per_pc = 3.086e18 #cm/pc
kb = 8.617e-5 #eV/K
T0_default = 5500 #K
deltaT_default = 1000 #K

def insert_diag(A, v, offset):
    rng = np.arange(len(v))
    if offset<0:
        rng = rng-offset
    A[rng, rng+offset] = v
    
def tridiag_plus(a, b, c, d, e):
    N = int(b.size - e.size)
    n = int(b.size/N)
    res = np.zeros(N*n)
    dp = np.zeros(N)
    cp = np.zeros(N)
    #start at last block
    for i in range(0, n):
        dp[0] = d[N*(n-1-i)]/b[N*(n-1-i)]
        cp[0] = c[N*(n-1-i)]/b[N*(n-1-i)]
        for j in range(1,N):
            cp[j] = c[N*(n-1-i)+j]/(b[N*(n-1-i)+j] - a[N*(n-1-i)+j]*cp[j-1])
            dp[j] = (d[N*(n-1-i)+j]-a[N*(n-1-i)+j]*dp[j-1])/(b[N*(n-1-i)+j]-a[N*(n-1-i)+j]*cp[j-1])
        if i == 0:
            res[N*n-1] = dp[N-1]
            for j in range(1,N):
                res[N*n-1-j] = dp[N-1-j]-cp[N-1-j]*res[N*n-j]
        else:
            ep = np.zeros((N,N))
            ep_v = np.zeros(N)
            ep_v[0] = e[N*(n-1-i)]/b[N*(n-1-i)]
            for j in range(1,N):
                ep_v[j] = e[N*(n-1-i)+j]/(b[N*(n-1-i)+j] - a[N*(n-1-i)+j]*cp[j-1])
            insert_diag(ep, ep_v, 0)
            for k in range(2, N+1):
                ep_v = ep_v[0:-1]
                ep_v = -a[N*(n-1-i)+k-1: N*(n-i)]*ep_v/(b[N*(n-1-i)+k-1: N*(n-i)]- a[N*(n-1-i)+k-1: N*(n-i)]*cp[k-2:-1])
                insert_diag(ep, ep_v, -k+1)
            res[N*(n-i)-1] = dp[-1] - sum(ep[-1]*res[N*(n-i): N*(n+1-i)])
            for j in range(1,N):
                res[N*(n-i)-1-j] = dp[-1-j] - sum(ep[-1-j]*res[N*(n-i): N*(n+1-i)]) - cp[-1-j]*res[N*(n-i)-j]
    return res

#radial function
#models = [[model0, B0, r0], [model1, B1, r1], ...]
def dist(r, models):
    res = 0
    for row in models:
        if row[0] == 'exp':
            res =  res + row[1]*np.exp(-r/row[2])
        elif row[0] == 'nfw':
            rho0 = row[1]
            gamma = row[2]
            r0 = row[3]
            res = res + rho0/(((r/r0)**gamma)*(1+r/r0)**(3-gamma))
        else:
            print('Error: model not defined')           
    return res

#derivative of the radial function
#models = [[model0, B0, r0], [model1, B1, r1], ...]
def ddistdr(r, models):
    res = 0
    for row in models:
        if row[0] == 'exp':
            res =  res -(row[1]/row[2])*np.exp(-r/row[2])
        else:
            print('Error: model not identified')           
    return res

#input: Energy in eV
#output: phasespace density of CMB photons at the energy of interest
def f_CMB(E):
    if E > 300*T_CMB:
        f = 0
    else:
        f = (np.pi**(-2)*E**2/((h_bar*c)**3))*(1/(np.exp(E/T_CMB)-1))
    return f

#input--> T: temperature in eV, E: Energy in eV, params=[T0, delta_T]
def T_dist(T, E, params=[T0_default*kb, deltaT_default*kb]):
    T0=params[0]
    delta_T = params[1]
    if T < E/300 or T==0:
        return 0
    else:
        return (E**2)*np.exp(-(T-T0)**2/(2*delta_T**2))/(np.exp(E/T)-1)

def ic_integrand(integ_vars, indep_vars, funcs):
    y = integ_vars[0]
    gamma = integ_vars[1]
    l = integ_vars[2]
    nu = indep_vars[0]
    E_ic = 2*np.pi*h_bar*nu
    rho = indep_vars[1]
    fe = funcs[0]
    f_star = funcs[1]
    f_CMB = funcs[2]
    f = (3*me*sigma_T*c/16*np.pi)*((f_star(E_ic/(4*gamma**2*y), np.sqrt(rho**2+l**2))+f_CMB(E_ic/(4*gamma**2*y)))*fe(np.sqrt(l**2+rho**2), gamma*me)/gamma**2)*(2*(1+1/y)-2*y+np.log(y))
    return f

def sync_integrand(integ_vars, indep_vars, funcs):
    y = integ_vars[0]
    alpha = integ_vars[1]
    gamma = integ_vars[2]
    l = integ_vars[3]
    nu = indep_vars[0]
    omega = 2*np.pi*nu
    rho = indep_vars[1]
    B = funcs[0]
    fe = funcs[1]
    me_g = me*g_per_GeV
    condlist = [np.logical_or(np.sin(alpha)==0, y==0), np.logical_and(np.logical_not(np.sin(alpha)==0), np.logical_not(y==0))]
    choicelist = [0, 2*np.pi*np.sqrt(3)*e*(me*g_per_GeV)*me*nu**2/(9*B(np.sqrt(rho**2+l**2))*gamma**4*y**2)*fe(np.sqrt(rho**2+l**2), me*gamma)*kv(5/3, 4*np.pi*nu*me*g_per_GeV*c/(3*e*B(np.sqrt(rho**2+l**2))*gamma**2*np.sin(alpha)*y))]
    return np.select(condlist, choicelist)

