# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:06:26 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np
from time import time

# Part 1 functions
def crosscorr(f, g):
    return np.dot(f - np.mean(f), g - np.mean(g))/len(f)

def norm_crosscorr(f, g):
    pass

def norm_crosscorr2d(t, A):
    pass
