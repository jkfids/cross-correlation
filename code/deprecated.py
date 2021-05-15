# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:50:42 2021

@author: Fidel
"""
# Import
import numpy as np

# Deprecated functions
def crosscorr(f, g):
    """
    Takes two vectors of the same size, subtracts the vector elements by their
    respective means, and passes one over the other to construct a 
    cross-correlation vector
    """
    N = len(f)
    r = np.array([], dtype=np.single)
    r1 = np.array([], dtype=np.single)
    r2 = np.array([], dtype=np.single)
    f = f - np.mean(f)
    g = g - np.mean(g)
    
    for i in range(N-1):
        r1i = np.dot(f[N-i-1:N], g[0:i+1])
        r2i = np.dot(f[0:N-i-1], g[i+1:N])
        r1 = np.append(r1, r1i)
        r2 = np.append(r2, r2i)
        
    r = np.append(r, r1)
    r = np.append(r, np.dot(f, g))
    r = np.append(r, r2)
    return r/N

