# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:26:59 2018

@author: e0008730
"""

import numpy as np
import types
import math

import matplotlib.pyplot as plt
import pandas as pd

import QuantumEvolution as qe

class SpinChain(qe.QuantumEvolution):
    def __init__(self, num_spin : int, N_t:int, tau:float, f:types.FunctionType = None, J:float=1.):
        qe.QuantumEvolution.__init__(self, 2**num_spin, N_t, tau)
        self._num_spin = num_spin
        self._J = J
        self.__prepare_matrices()
        #uniform external B field, could be time dependent
        self._B_uniform = f
        if self._B_uniform == None:
            print('the external B field is not defined!')
            print('A constant field is applied.')
            #x here is necessary for the function to accept a variable
            self._B_uniform = lambda x: np.array([0.5, 0.5, 0.5])
            
    def __prepare_matrices(self):
        self.identity = np.array([[1.0+0j, 0+0j],[0+0j, 1.0+0j]])
        #Pauli matrices
        self.sigma = np.zeros((3,2,2),dtype=np.complex_)
        self.sigma[0] = np.array([[0, 1],[ 1, 0]])
        self.sigma[1] = np.array([[0, -1j],[1j, 0]])
        self.sigma[2] = np.array([[1, 0],[0, -1]])
        
        #coupling term
        self.coupling = np.zeros((self._dim,self._dim),dtype=np.complex_)
        for i in range(self._num_spin-1):
            temp = np.matrix(1.)
            for j in range(self._num_spin):
                if j==i or j==i+1:
                    temp = qe.my_outer(temp, self.sigma[2])
                else:
                    temp = qe.my_outer(temp, self.identity)
            self.coupling = self.coupling - self._J*temp
        
        #uniform external field
        self.S_total = np.zeros((3, self._dim, self._dim),dtype=np.complex_)
        for i in range(3):
            for k in range(self._num_spin):
                temp = np.matrix(1.)
                for j in range(self._num_spin):
                    if j==k:
                        temp = qe.my_outer(temp, self.sigma[i])
                    else:
                        temp = qe.my_outer(temp, self.identity)
                self.S_total[i] += temp
    
    #could override, or simply pass a function to define the external B field
    def B(self, i:int):
        return self._B_uniform(i)
    
    
    def Hamiltonian(self, i : int):
        temp = self.coupling.copy()
        temp -= np.tensordot(self.B(i), self.S_total, axes = 1)
        #plus or minus?
        #temp += np.tensordot(self.B(i), self.S_total, axes = 1)
        return temp
        

if __name__ == '__main__':
    

    N_iters = 30
    tau_min = 100
    tau_max = 100000
    N_t_min = 5000
    dt = 0.1
    
    
    beta_hot = 0.5
    beta_cold = 1.0
    
    
    taus = np.array(range(N_iters + 1))
    taus = math.log(tau_max/tau_min)*taus/N_iters
    taus = tau_min*np.exp(taus)
    Ws=[]
    count = 0
    for tau in taus:
        print('%d / %d  running' % (count,len(taus)),end='\r')
        count=count+1
        
        spins = 5
        N_t = int(tau/dt)
        if N_t < N_t_min:
            N_t = N_t_min
        
        def B_t(i:int):
            #Bx = 0 will produce error. why?
            #change z component
            Bx = 0.2
            By = 0.2
            Bz = 0.5*np.sin((np.pi/2.0)*i/N_t)
#            #change y component
#            Bx = 0.1
#            By = 0.5*np.sin((np.pi/2.0)*i/N_t)
#            Bz = 0.2
            return np.array([Bx, By, Bz])

        spinch = SpinChain(spins, N_t, tau, B_t)
    
    #    norm = np.array([1,1,1])
    #    print(np.tensordot(norm, a.sigma, axes = 1))
        
        H_0 = spinch.Hamiltonian(0)
        values_0, vectors_0 = np.linalg.eig(H_0)
        
    #    #verify that vectors_0 are eigen column vectors
    #    D = np.diag(values_0)
    #    B = np.dot(np.dot(vectors_0,D), vectors_0.conj().T)
    #    print(B-H_0)
        init = vectors_0.copy()
        final = spinch.time_evolution(init)
        
        H_tau = spinch.Hamiltonian(N_t)
        values_tau, vectors_tau = np.linalg.eig(H_tau)
        
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
        
        beta = 1.0
        p_0 = np.exp(-beta * values_0.real)
        Z_0 = np.sum(p_0)
        F_0 = -beta*math.log(Z_0)
        p_0 = p_0/Z_0
        
        
        p_tau = np.exp(-beta * values_tau.real)
        Z_tau = np.sum(p_tau)
        F_tau = -beta*math.log(Z_tau)
        p_tau = p_tau/Z_tau
        
        #to_measure = np.exp(-beta*W.T)
        to_measure = W.T
        
        Ws.append(np.dot(to_measure*Transition, p_0).sum())
        error = np.dot(np.exp(-beta*W.T)*Transition, p_0).sum()-math.exp(-beta*(F_tau - F_0))
        if error > 1e-10:
            print('error=',error,'>1e-10')
    
    a = pd.DataFrame({'taus': taus, 'mean_W': Ws})
    a.to_csv('average_W_z.csv')
    
    fig, ax = plt.subplots()
    
    ax.semilogx(taus, Ws, 'o')
    #ax.semilogx(tau, sigma, 'x')
    ax.grid()
    #plt.savefig('test.png')
    plt.show()
    
    #print(Transition)
    #print(values_0)
#    print(canonical.canonical_energy(10.0, values_0))
#    print(canonical.canonical_energy(0.5, values_0))

    
        
