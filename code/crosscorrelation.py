# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:06:26 2021

@author: Fidel
"""

# Standard libraries
import numpy as np
from numpy import mean, sqrt, dot, conjugate
from scipy.fft import fft, ifft
from numba import njit

# Part 1 functions


@njit
def standev(f):
    """Calculate the standard deviation of an input vector"""
    f = f - mean(f)
    N = len(f)
    return sqrt(dot(f, f)/N)


@njit
def crosscorr(f, g):
    """
    Takes two numpy arrays of the same size and passes one over the other to 
    construct a cross-correlation vector
    """
    N = len(f)
    r = np.zeros(2*N - 1, dtype=np.float64)
    r[N-1] = dot(f, g)
    for i in range(N-1):
        # Calculate elements of the cross-correlation vector by taking dot
        # products of (input) vector slices
        r[i] = dot(f[0:i+1], g[N-1-i:N])
        r[N+i] = dot(f[i+1:N], g[0:N-1-i])
    return r


@njit
def norm_crosscorr(f, g):
    """"
    Fully normalised version of crosscorr that subtracts the vector by its
    mean first then divides the correlation vector by the product of standard 
    deviations
    """
    f = f - mean(f)
    g = g - mean(g)
    N = len(f)
    return crosscorr(f, g)/(N*standev(f)*standev(g))
    # return crosscorr(f, g)/(np.std(f)*np.std(g))


@njit
def norm_crosscorr2d(t, A):
    """
    Calculate the normalized cross-correlation between template matrix t
    and search region matrix A
    """
    # Calculate the width and height of the cross-correlation matrix
    h_A, w_A = np.shape(A)
    h_t, w_t = np.shape(t)
    h_R = h_A - h_t + 1
    w_R = w_A - w_t + 1
    # Initialise R matrix
    R = np.zeros((h_R, w_R))  
    t = t - mean(t)
    sigma_t = sqrt(np.sum(t*t))
    if sigma_t == 0:  # In case of division by 0 (sigma_t)
        return R
    # Slide t over A via nested for loops, normalising at each step
    for i in range(h_R):
        for j in range(w_R):
            A_subset = A[i:i+h_t, j:j+w_t]
            A_subset = A_subset - mean(A_subset)
            sigma_A = sqrt(np.sum(A_subset*A_subset))
            if sigma_A != 0:  # In case of division by 0 (sigma_A)
                R[i, j] = np.sum(A_subset*t)/(sigma_A*sigma_t)
    return R


def spectral_crosscorr(f, g):
    """
    Calculate the cross-correlation matrix via Scipy's fast fourier transform
    """
    return ifft(conjugate(fft(f))*fft(g)).real


@njit
def calc_offset(R, scale, mode='spatial'):
    """
    Calculate the time offset between two signals given their cross-correlation 
    vector and time scale (sampling rate)
    """
    if mode == 'spatial':
        len_f = (len(R)+1)/2  # Length of input vector
        return (len_f - 1 - np.argmax(R))*scale
    elif mode == 'spectral':
        return (np.argmax(R) - len(R))*scale
