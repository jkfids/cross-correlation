# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:14:46 2021

@author: Fidel
"""

# Standard libraries
import numpy as np
from numpy import exp

# Local modules
from crosscorrelation import *

def gaussian(x, A, sigma):
    return A*exp(-(x)**2/(2*sigma*sigma))

def gaussian2d(x, y, A, sigma):
    return A*exp(-(x*x+y*y)/2*sigma*sigma)