# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:04:29 2021

@author: Fidel
"""

# Standard libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from time import time

# Local modules
from crosscorrelation import norm_crosscorr2d

def gaussian(x, a, sigma):
    return a*np.exp(-x*x/(2*sigma*sigma))

def gaussian2d(x, y, a, sigma):
    return a*np.exp(-(x*x+y*y)/(2*sigma*sigma))

search = Image.open('data/cal_image_left_2000.tiff')
search = np.array(search.convert('L'))

# Using curve_fit from scipy, determine the parameters of the gaussian model
dot = search[171:207,384:420] # Isolate a single dot
row_max = np.argmax(np.mean(dot, axis=0))
dot_slice = dot[row_max,:]
x_dot_slice = np.linspace(-17.5, 17.5, len(dot_slice))
(a, sigma), _ = curve_fit(gaussian, x_dot_slice, dot_slice)

# Generate dot template
L = 8 # Length of square template
template = np.zeros((L,L))
template_slice = np.linspace(-L/2+0.5, L/2-0.5, L)
for i in range(L):
    template[i] = gaussian2d(template_slice, i-(L-1)/2, a, sigma)
    
# Calculate 2d cross correlation matrix
start = time()
R = norm_crosscorr2d(template, search)
end = time()
print(f'Time elapsed (calculate 2d cross-corr vector): {round(end - start, 2)}s')

