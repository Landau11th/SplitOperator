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

class QuantumEvolution:
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
        D = np.diag(np.exp((0-1j)*d*self._dt))
        dU = np.dot(Eig, np.dot(D, Eig.conj().T))
        return np.dot(dU,init_state)
        
    #time evolution from 0 to tau
    def time_evolution(self, init_state : np.array):
        phi = np.copy(init_state) 
        for i in range(self._N_t):
            phi = self.one_step(phi, i)
        return phi

#to do proper outer product and convert to 2D matrix
def my_outer(A:np.matrix, B:np.matrix):
    temp = np.multiply.outer(A,B)    
    blk = []
    for i in range(A.shape[0]):
        ls = []
        for j in range(A.shape[1]):
            ls.append(temp[i,j])
        blk.append(ls)    
    return (np.block(blk))
    
        
