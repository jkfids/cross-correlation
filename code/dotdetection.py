# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:04:29 2021

@author: Fidel
"""

# Standard libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from time import time

# Local modules
from crosscorrelation import norm_crosscorr2d
from stereovision import gaussian, gaussian2d, Calibration

# Load calibration file and convert into grayscale matrix
search = Image.open('data/cal_image_left_2000.tiff')
search = np.array(search.convert('L'))

# Use scipy.optimize to fit a gaussian to a single calibration dot and 
# determine the ideal gaussian template parameters
dot = search[171:207,384:420] # Isolate a single dot
row_max = np.argmax(np.mean(dot, axis=0))
dot_slice = dot[row_max,:] # Take 1D cross-section of the dot
x_dot_slice = np.linspace(-17.5, 17.5, len(dot_slice))
(a, sigma), _ = curve_fit(gaussian, x_dot_slice, dot_slice)
print(f'Ideal gaussian parameters: a = {a}, sigma = {abs(sigma)}')

# Generate the dot template using previously found parameters
L = 12 # Length of square template
template = np.zeros((L, L))
template_row = np.linspace(-L/2+0.5, L/2-0.5, L)
for i in range(L):
    template[i] = gaussian2d(template_row, i-(L-1)/2, a, sigma)
    
# Calculate the 2d cross correlation matrix
start = time()
R = norm_crosscorr2d(template, search)
end = time()
print(f'Time elapsed (calculate 2d cross-corr vector): {round(end - start, 2)}s')
    
# Convert and filter cross-correlation matrix into an ordered array of 
# coordinates where each dot corresponds to a single coordinate
start = time()
coords = Calibration.filter_coords(R, L)
end = time()
print(f'Time elapsed (filter coordinates): {round(end - start, 2)}s')
plt.imshow(search)
plt.scatter(coords[:,0], coords[:,1], s=60, facecolors='none', edgecolors='y')
plt.savefig('output/dotdetection.png', dpi=144)
