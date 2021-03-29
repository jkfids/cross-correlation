# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:06:26 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np

# Part 1 functions
def crosscorr(f, g):
    return np.correlate(f, g, 'full')

def norm_crosscorr(f, g):
    pass