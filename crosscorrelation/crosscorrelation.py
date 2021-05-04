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
def standev(f):
    """Calculate the standard deviation of an input vector"""
    f = f - np.mean(f)
    N = len(f)
    return np.sqrt(np.dot(f, f)/N)

@njit
def crosscorr(f, g):
    """
    Takes two numpy arrays of the same size and passes one over the other to 
    construct a cross-correlation vector
    """
    N = len(f)
    r = np.zeros(2*N - 1, dtype=np.single)
    r[N-1] = np.dot(f, g)
    for i in range(N-1):
    # Calculate elements of the cross-correlation vector by taking dot
    # products of (input) vector slices
        r[i] = np.dot(f[0:i+1], g[N-1-i:N])
        r[N+i] = np.dot(f[i+1:N], g[0:N-1-i])
    return r

@njit
def norm_crosscorr(f, g):
    """"
    Fully normalised version of crosscorr that subtracts the vector by its
    mean first then divides the correlation vector by the product of standard 
    deviations
    """
    f = f - np.mean(f)
    g = g - np.mean(g)
    N = len(f)
    return crosscorr(f, g)/(N*standev(f)*standev(g))
    #return crosscorr(f, g)/(np.std(f)*np.std(g))

@njit
def norm_crosscorr2d(t, A):
    """
    Calculate the normalized cross-correlation between template matrix t
    and search region matrix A
    """
    A_h, A_w = np.shape(A)
    t_h, t_w = np.shape(t)
    # Calculate the width and height of the cross-correlation matrix
    R_h = A_h - t_h + 1 
    R_w = A_w - t_w + 1 
    #R = np.zeros([R_h, R_w])
    R = np.zeros((R_h, R_w)) # @njit version of np.zeros inputs tuple
    t = t - np.mean(t)
    # Slide t over A via nested for loops, normalising at each step
    for i in range(R_h):
        for j in range(R_w):
            A_subset = A[i:i+t_h, j:j+t_w]
            A_subset = A_subset - np.mean(A_subset)
            sigma_A = np.sqrt(np.sum(A_subset**2))
            R[i, j] = np.sum(A_subset*t)/sigma_A
    sigma_t = np.sqrt(np.sum(t**2))
    R = R/sigma_t
    return R

def spectral_crosscorr(f, g):
    """
    Calculate the cross-correlation matrix using Scipy fast fourier transform
    """
    return ifft(np.conjugate(fft(f))*fft(g)).real
    
@njit
def calc_offset(R, scale):
    """
    Calculate the time offset between two signals given their cross-correlation 
    vector
    """
    len_f = (len(R)+1)/2 # Length of input vector
    return (len_f - 1 - np.argmax(R))*scale

@njit
def calc_fourier_offset(R, scale):
    """
    Calculate the time offset between two signals given their spectral
    cross-correlation vector
    """
    return (np.argmax(R) - len(R))*scale
