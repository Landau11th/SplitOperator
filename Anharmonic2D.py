# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 13:52:10 2018

@author: e0008730
"""
import math
from QuantumEvolution import SplitOperator
import numpy as np
import time

class Anharmonic2D(SplitOperator):
    _hbar = 1.0
    #hbar is taken to be 1
    lmbd = 0.05
    omega_x = 1.0
    omega_y_0 = 1.0
    omega_y_tau = 1.25
    beta = 1.0
    _L = 10.0*math.sqrt(omega_y_0)
    
    def __init__(self, matrix_dim:int, N_t:int, tau:float):
        SplitOperator.__init__(self, matrix_dim, N_t, tau)  
        self._potential_H_prepare()
        self._kinetic_H_prepare()
        pass
    
    def _kinetic_H_prepare(self):
        self._dp = 2*math.pi/self._L*self._hbar
        temp = []
        for i in range(self._dim):
            if i <= self._dim/2:
                temp.append(i*self._dp)
            else:
                temp.append( (i-self._dim)*self._dp)
        p = np.array(temp)
        p = p*p
        K = np.reshape(p.repeat(self._dim), (self._dim,self._dim))
        self.T_fft = 0.5*( K+K.T )
        #to generate eigenstates
        temp = np.array(range(self._dim))
        row = np.reshape(temp.repeat(self._dim), (self._dim,self._dim))
        col = row.T.copy()
        iiprime = row - col
        #first = 0.5*(-1)**iiprime/self._dx**2
        first = np.vectorize(lambda x: -1 if x%2 == 1 else 1)(iiprime)
        first = (0.5/self._dx**2)*first
        second = iiprime**2/2 + (3/np.pi**2)*np.identity(self._dim)
        second = 1.0/second
        #print(second)
        self.eig_momentum2 = first*second
    
    def kinetic_H(self, i:int):
        return self.T_fft
    
    def _potential_H_prepare(self):
        self._dx = self._L/self._dim
        x = np.array(range(self._dim))*self._L/self._dim + ( self._dx/2 - self._L/2 )
        y = x.copy()
        #first index for x, second for y
        self.V_x2 = np.reshape((x*x).repeat(self._dim), (self._dim,self._dim)).T
        self.V_y2 = np.reshape((y*y).repeat(self._dim), (self._dim,self._dim)).T
        self.V_lmbd = self.V_x2*self.V_y2
        #to generate eigenstates
        self.eig_coord = x.copy() 
    
    def omega_y(self, i:int):
        return self.omega_y_0 + (self.omega_y_tau-self.omega_y_0)*i/self._N_t
    def lmbd_t(self, i:int):
        #return self.lmbd
        return self.lmbd*(1-i/self._N_t)*i/self._N_t
    
    
    def potential_H(self, i:int):
        temp = (0.5*self.omega_x**2) * self.V_x2
        temp += (0.5*self.omega_y(i)**2) * self.V_y2
        temp += self.lmbd_t(i) * self.V_lmbd
        return temp
    
    def one_step(self, init_state, i:int):
        evlv = init_state*np.exp( (-1j*self._dt)*self.potential_H(i) )
        evlv = np.fft.fft2(evlv)
        evlv = evlv*np.exp( (-1j*self._dt)*self.kinetic_H(i) )
        return np.fft.ifft2(evlv)
   
    
if __name__ == "__main__":
    dim = 256
    N_t = 1000
    tau = 1.0
    
    anha = Anharmonic2D(dim, N_t, tau)
    
    
    Hx = anha.eig_momentum2 + np.diagflat(0.5*(anha.omega_x**2)*(anha.eig_coord**2))
    #print(Hx)#+ np.diagflat(0.5*(anha.omega_x**2)*(anha.eig_coord**2))
    eig_x, v_x = np.linalg.eigh(Hx)
    #print(sorted(eig_x))
    Hx = anha.eig_momentum2 + np.diagflat(0.5*(anha.omega_x**2)*(anha.eig_coord**2))
    #print(Hx)#+ np.diagflat(0.5*(anha.omega_x**2)*(anha.eig_coord**2))
    eig_x, v_x = np.linalg.eigh(Hx)
    
    phi_init = np.zeros((dim,dim), dtype=np.complex64)
    
    start_time = time.time()
    
    #print(anha.time_evolution(phi_init))
    
    elapsed_time = time.time() - start_time
    print(elapsed_time)
        