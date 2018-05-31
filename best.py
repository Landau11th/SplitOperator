# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:57:04 2018

@author: e0008730
"""
import numpy as np
import types
import math

import matplotlib.pyplot as plt
import pandas as pd
import SplitOperatorMethod as SOM


def ensemble(beta:float, spins:int, N_t:int, tau:float, B_t_dep:types.FunctionType):
    spinch = SOM.SpinChain(spins, N_t, tau, B_t_dep)
    #eigenstates
    H_0 = spinch.Hamiltonian(0)
    values_0, vectors_0 = np.linalg.eig(H_0)
    H_tau = spinch.Hamiltonian(N_t)
    values_tau, vectors_tau = np.linalg.eig(H_tau)
    #evolve states
    init = vectors_0.copy()
    final = spinch.time_evolution(init)

    #Amp_i->j = final_i dot vector_tau_j
    Amp = np.dot(final.conj().T, vectors_tau)
    Transition = np.absolute(Amp)
    Transition = Transition*Transition

    dim = len(values_0)
    E_0 = values_0.real
    E_0 = np.reshape(np.repeat(E_0,dim), (dim,dim))
    E_tau = values_tau.real
    E_tau = np.reshape(np.repeat(E_tau,dim), (dim,dim))
    #W_i->j = E_tau_j - E_0_i
    W = E_tau.T - E_0
    
    p_0 = np.exp(-beta*values_0.real)
    Z_0 = np.sum(p_0)
    F_0 = -(1.0/beta)*math.log(Z_0)
    p_0 = p_0/Z_0
    
    p_tau = np.exp(-beta*values_tau.real)
    Z_tau = np.sum(p_tau)
    F_tau = -(1.0/beta)*math.log(Z_tau)
    p_tau = p_tau/Z_tau
    
    #to_measure = np.exp(-beta*W.T)
    to_measure = W.T
    #print(B_t_dep.__name__+': ',(F_0-F_tau))
    error = np.dot(np.exp(-beta*W.T)*Transition, p_0).sum()-math.exp(-beta*(F_tau - F_0))
    if abs(error) > 1e-8:
        print('tau=',tau,B_t_dep.__name__)
        print('JE error=',error,'>1e-8')
    
    #return the avrerage of quantity we wish to calculate
    return np.dot(to_measure*Transition, p_0).sum()


if __name__ =='__main__':
    N_iters = 40 #40
    tau_min = 0.1#10000#100
    tau_max = 1000#100000000#1000000
    N_t_min = 2000
    N_t_max = 2000000 #2000000
    dt = 0.01
    
    beta_hot = 0.5
    beta_cold = 1.0
    
    taus = np.array(range(N_iters + 1))
    taus = math.log(tau_max/tau_min)*taus/N_iters
    taus = tau_min*np.exp(taus)
    
    Ws_forward=[]
    Ws_backward=[]
    
    count = 0
    for tau in taus:
        print('running ', count,'/', N_iters,end='\r')
        count=count+1
        
        spins = 4#5
        N_t = int(tau/dt)
        if N_t < N_t_min:
            N_t = N_t_min
        elif N_t > N_t_max:
            N_t = N_t_max
        
        def B_t(i:int):
            #Bx = 0 will produce error. why?
            #change z component
            #Bx = 0.2
            #By = 0.2
            #Bz = 0.4*float(i)/N_t + 0.1
            Bx = 0.3
            By = 0.3
            #Bz = 0.5*(1.0 - float(i)/N_t)
            #Bz = 0.5*(0.2+float(i)/N_t)
            Bz = 0.5*math.sin(math.pi/2.0*float(i)/N_t) + 0.3
            #Bz = 0.5*np.sin((np.pi/2.0)*i/N_t)
#            #change y component
#            Bx = 0.1
#            By = 0.5*float(i)/N_t
#            Bz = 0.2
            return np.array([Bx, By, Bz])
        
        def B_t_revert(i:int):
            return B_t(N_t-i)
    
        Ws_forward.append(ensemble(beta_cold, spins, N_t, tau, B_t))
        Ws_backward.append(ensemble(beta_hot , spins, N_t, tau, B_t_revert))
    
    #savefile to csv
    a = pd.DataFrame({'taus': taus, 'mean_W_forward': Ws_forward, 'mean_W_backward':Ws_backward})
    a.to_csv('average_W.csv')
    
    fig, ax = plt.subplots()
    
    ax.semilogx(taus, Ws_forward,'bo', taus, Ws_backward, 'ro')
    #ax.semilogx(tau, sigma, 'x')
    ax.grid()
    #plt.savefig('test.png')
    plt.show()
    
    