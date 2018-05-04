# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:26:59 2018

@author: e0008730
"""

import numpy as np

def canonical_energy(beta:float, eigen_E:np.array):
    temp = np.exp(-beta * eigen_E)
    Z = np.sum(temp)
    temp = temp*eigen_E
    return np.sum(temp)/Z
    

