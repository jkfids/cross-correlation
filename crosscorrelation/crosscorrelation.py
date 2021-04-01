# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:06:26 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np
#from numba import jit


# Part 1 functions
#@jit(nopython=True)
def crosscorr(f, g):
    """
    Takes two vectors of the same size and passes one over the other to create
    a correlation vector
    """
    f = np.array(f) - np.mean(f)
    g = np.array(g) - np.mean(g)
    N = len(f)
    r = np.zeros(2*N - 1, dtype=np.single)
    
    r[N-1] = np.dot(f, g)
    for i in range(N-1):
        r[i] = np.dot(f[0:i+1], g[N-1-i:N])
        r[N+i] = np.dot(f[i+1:N], g[0:N-1-i])
    return r/N

def norm_crosscorr(f, g):
    """"
    Normalised version of crosscorr that divides the correlation vector by
    a product of the input vectors' standard deviations
    """
    return crosscorr(f, g)/(np.std(f)*np.std(g))

def norm_crosscorr2d(t, A):
    pass

def get_roll(R):
    return round(len(R)/2-0.5-np.argmax(R))

if __name__ == "__main__":
    f = list(range(100))
    r = crosscorr(f,f)