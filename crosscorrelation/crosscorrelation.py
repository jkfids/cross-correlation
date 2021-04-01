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
    Takes two vectors of the same size, subtracts the vector elements by their
    respective means, and passes one over the other to construct a 
    cross-correlation vector
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
    #return crosscorr(f, g)/(standev(f)*standev(g))
    return crosscorr(f, g)/(np.std(f)*np.std(g))

def norm_crosscorr2d(t, A):
    """
    Calculate the normalized cross-correlation between template matrix t
    and search region matrix A
    """
    pass

def standev(f):
    """Calculate the standard deviation of an input vector"""
    f = np.array(f) - np.mean(f)
    N = len(f)
    return np.sqrt(np.sum(f**2)/N)
    
def calc_offset(R, scale):
    """
    Calculate the time offset between two signals given their cross-correlation 
    vector
    """
    len_f = (len(R)+1)/2 # Length of input vector
    return (len_f - 1 - np.argmax(R))*scale
