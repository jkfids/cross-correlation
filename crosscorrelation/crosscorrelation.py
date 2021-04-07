# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:06:26 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np
from scipy.fft import fft, ifft
from numba import njit

# Part 1 functions
@njit
def crosscorr(f, g):
    """
    Takes two vectors of the same size, subtracts the vector elements by their
    respective means, and passes one over the other to construct a zero
    normalised cross-correlation vector
    """
    f = f - np.mean(f)
    g = g - np.mean(g)
    N = len(f)
    r = np.zeros(2*N - 1, dtype=np.single)
    
    r[N-1] = np.dot(f, g)
    for i in range(N-1):
        r[i] = np.dot(f[0:i+1], g[N-1-i:N])
        r[N+i] = np.dot(f[i+1:N], g[0:N-1-i])
    r = r/N
    return r

@njit
def norm_crosscorr(f, g):
    """"
    Fully normalised version of crosscorr that divides the correlation vector
    by a product of the input vectors' standard deviations
    """
    return crosscorr(f, g)/(standev(f)*standev(g))
    #return crosscorr(f, g)/(np.std(f)*np.std(g))

@njit
def norm_crosscorr2d(t, A):
    """
    Calculate the normalized cross-correlation between template matrix t
    and search region matrix A
    """
    t = t - np.mean(t)
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
    #R = np.zeros([R_h, R_w])
    R = np.zeros((R_h, R_w)) # @njit version of np.zeros inputs tuple, not list
    for i in range(R_h):
        for j in range(R_w):
            A_subset = A[i:i+t_h, j:j+t_w]
            A_subset = A_subset - np.mean(A_subset)
            sigma_A = np.sqrt(np.sum(A_subset**2))
            R[i, j] = np.sum(A_subset*t)/sigma_A
            #print(f'{i*R_w+j}/{round(R_w*R_h)}')
    sigma_t = np.sqrt(np.sum(t**2))
    R = R/sigma_t
    return R

def spectral_crosscorr(f, g):
    return ifft(np.conjugate(fft(f))*fft(g))

@njit
def standev(f):
    """Calculate the standard deviation of an input vector"""
    f = f - np.mean(f)
    N = len(f)
    return np.sqrt(np.sum(f**2)/N)
    
@njit
def calc_offset(R, scale):
    """
    Calculate the time offset between two signals given their cross-correlation 
    vector
    """
    len_f = (len(R)+1)/2 # Length of input vector
    return (len_f - 1 - np.argmax(R))*scale
