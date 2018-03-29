# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:26:59 2018

@author: e0008730
"""

import numpy as np
import canonical

class SplitOperator:
    def __init__(self, matrix_dim:int, N_t:int, tau:float):
        self._dim = matrix_dim
        self._N_t = N_t
        self._tau = tau
        self._dt = tau/N_t
    
    #returns the Hamiltonian at specific time t=i*self._dt
    def Hamiltonian(self, i:int):
        H = np.zeros((self._dim, self._dim), dtype=np.complex_)
        raise TypeError('Hamiltonian must be defined and return values!')
        return H
    
    #let the state evolve time period of dt
    def one_step(self, init_state:np.array, i:int):
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
            
def my_outer(A:np.matrix, B:np.matrix):
    temp = np.multiply.outer(A,B)    
    blk = []
    for i in range(A.shape[0]):
        ls = []
        for j in range(A.shape[1]):
            ls.append(temp[i,j])
        blk.append(ls)    
    return (np.block(blk))
    

class SpinChain(SplitOperator):
    def __init__(self, num_spin : int, N_t:int, tau:float, J:float=1.):
        SplitOperator.__init__(self, 2**num_spin, N_t, tau)
        self._num_spin = num_spin
        self._J = J
        self.__prepare_matrices()
            
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
                    temp = my_outer(temp, self.sigma[2])
                else:
                    temp = my_outer(temp, self.identity)
            self.coupling = self.coupling - self._J*temp
        
        #uniform external field
        self.uniform_B = np.zeros((3, self._dim, self._dim),dtype=np.complex_)
        for i in range(3):
            for k in range(self._num_spin):
                temp = np.matrix(1.)
                for j in range(self._num_spin):
                    if j==k:
                        temp = my_outer(temp, self.sigma[i])
                    else:
                        temp = my_outer(temp, self.identity)
                self.uniform_B[i] += temp
    
    def B(self, i:int):
        Bx = 1.0
        By = 1.0
        Bz = 0.5
        return np.array([Bx, By, Bz])
    
    def Hamiltonian(self, i : int):
        temp = self.coupling.copy()
        temp += np.tensordot(self.B(i), self.uniform_B, axes = 1)
        return temp
        
        
        
if __name__ == '__main__':
    
    spins = 5
    N_t = 2000
    tau = 1.0
        
    spinch = SpinChain(spins, N_t, tau)
    
#    norm = np.array([1,1,1])
#    print(np.tensordot(norm, a.sigma, axes = 1))
    
    H_0 = spinch.Hamiltonian(0)
    values_0, vectors_0 = np.linalg.eig(H_0)
    
#    D = np.diag(values_0)
#    B = np.dot(np.dot(vectors_0,D), vectors_0.conj().T)
#    print(B-H_0)
    
    
#    print(values_0[0])
#    
#    phi = vectors_0[:,0].copy()
#    print(np.dot(phi.conj(),phi))
#    
#    Hphi = np.dot(H,phi)
#    print(np.dot(Hphi.conj(),phi))
#
#    phi_tau = spinch.time_evolution(phi)
#    print(np.dot(phi_tau.conj(),phi_tau))
#    
#    print(np.dot(phi.conj(),phi_tau))
    
    init = vectors_0.copy()
    final = spinch.time_evolution(init)
    
    H_tau = spinch.Hamiltonian(N_t)
    values_tau, vectors_tau = np.linalg.eig(H_tau)
    
    Amp = np.dot(final.conj().T, vectors_tau)
    Transition = np.absolute(Amp)
    Transition = Transition*Transition
    print(Transition)
    
    print(canonical.canonical_energy(1.0, values_0))
    print(canonical.canonical_energy(0.5, values_0))

    
        
