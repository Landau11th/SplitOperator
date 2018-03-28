# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:26:59 2018

@author: e0008730
"""

import numpy as np

class SplitOperator:
    def __init__(self, matrix_dim : int, N_t : int, tau : float):
        self._dim = matrix_dim
        self._N_t = N_t
        self._tau = tau
        self._dt = tau/N_t
    
    #returns the Hamiltonian at specific time t=i*self._dt
    def Hamiltonian(self, i : int):
        H = np.zeros((self._dim, self._dim), dtype=np.complex_)
        raise TypeError('Hamiltonian must be defined and return values!')
        return H
    
    #let the state evolve time period of dt
    def one_step(self, init_state : np.array, i : int):
        #since H(t) = self.Hamiltonian(i)
        d, Eig = np.linalg.eig(self.Hamiltonian(i))
        D = np.diag(np.exp((0-1j)*d))
        dU = np.dot(Eig, np.dot(D, Eig.conj().T))
        return np.dot(dU,init_state)
        
    #time evolution from 0 to tau
    def time_evolution(self, init_state : np.array):
        phi = np.copy(init_state) 
        for i in range(self._N_t):
            phi = self.one_step(phi, i)
        return phi
            

class SpinChain(SplitOperator):
    def __init__(self, num_spin : int, N_t : int, tau : float):
        SplitOperator.__init__(self, 2**num_spin, N_t, tau)
        self._num_spin = num_spin
        #Pauli matrices
        self.sigma = np.zeros((3,2,2),dtype=np.complex_)
        self.sigma[0] = np.array([[0, 1],[ 1, 0]])
        self.sigma[1] = np.array([[0, -1j],[1j, 0]])
        self.sigma[2] = np.array([[1, 0],[0, -1]])        
    
    def Hamiltonian(self, i : int):
        pass

        
        
        
if __name__ == '__main__':
    
    print((1+2j)*(0+1j))
    
    a = SpinChain(5, 1000, 1.0)
    
    A = np.array([[ 2.,  1.0+1j,  3.],
               [ 1.0-1j,  1.,  1.],
               [ 3.,  1.,  4.]])
    
    w, V = np.linalg.eig(A)
    D = np.diag(w)
    B = np.dot(np.dot(V,D), V.conj().T)
    print(w)
    print(V)
    print(B)
    
    norm = np.array([1,1,1])
    print(np.tensordot(norm, a.sigma, axes = 1))
    
        
