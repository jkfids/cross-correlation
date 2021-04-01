# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:06:26 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np
from scipy import fft
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
    r = r/N
    return r

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
    t = np.array(t) - np.mean(t)
    A = np.array(A)
    sigma_A = 0
    try:
        A_h, A_w = np.shape(A)
        t_h, t_w = np.shape(t)
    except:
        A_h = np.shape(A)[0]
        t_h = np.shape(t)[0]
        A_w = A_h
        t_w = t_h
    R_h = A_h - t_h + 1 
    R_w = A_w - t_w + 1 
    R = np.zeros([R_h, R_w])
    for i in range(R_h):
        for j in range(R_w):
            A_subset = A[i:i+t_h, j:j+t_w]
            A_subset = A_subset - np.mean(A_subset)
            sigma_A += np.sum(A_subset**2)
            R[i, j] = np.sum(A_subset*t)
            print(f'{i*R_w+j}/{round(R_w*R_h)}')
    sigma_t = np.sum(t**2)
    R = R/np.sqrt(sigma_A*sigma_t)
    return R

def spectal_crosscorr():
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

if __name__ == "__main__":
    t = np.random.rand(2,2)
    A = np.random.rand(10,10)
    print(norm_crosscorr2d(t, A))   