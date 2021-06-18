# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:04:29 2021

@author: Fidel
"""

# Standard libraries
from time import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image


# Local modules
from crosscorrelation import norm_crosscorr2d
from calibration import gaussian, gaussian2d, Calibration

# Load calibration file and convert into grayscale matrix
search = Image.open('data/calibration/cal_image_left_2000.tiff')
search = np.array(search.convert('L'))

# Use scipy.optimize to fit a gaussian to a single calibration dot and
# determine the ideal gaussian template parameters
dot = search[171:207, 384:420]  # Isolate a single dot
row_max = np.argmax(np.mean(dot, axis=0))
dot_slice = dot[row_max, :]  # Take 1D cross-section of the dot
x_dot_slice = np.linspace(-17.5, 17.5, dot_slice.size)
(a, sigma), _ = curve_fit(gaussian, x_dot_slice, dot_slice)
print(f'Ideal gaussian parameters: a = {a}, sigma = {abs(sigma)}')

# Generate the dot template using previously found parameters
L = 8  # Length of square template
template = np.zeros((L, L))
template_row = np.linspace(-L/2+0.5, L/2-0.5, L)
for i in range(L):
    template[i] = gaussian2d(template_row, i-(L-1)/2, a, sigma)
    
# Construct plots of the gaussian template
X = np.arange(-18, 19, 1)
Y = np.arange(-18, 19, 1)
X, Y = np.meshgrid(X, Y)
Z = gaussian2d(X, Y, a, sigma)

fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, 
                                dpi=144, figsize=(6,6))
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', antialiased=False)
ax1.set_zlabel('Pixel Intensity')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlim(0, 250)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
fig1.tight_layout()
fig1.savefig('output/gaussian2dplot.png')

fig2, ax2 = plt.subplots(dpi=144, figsize=(6,6))
heatmap = ax2.imshow(Z, extent=(-18,18,-18,18))
ax2.set_xlabel('x (pixels)')
ax2.set_ylabel('y (pixels)')
fig2.colorbar(heatmap, fraction=0.046, pad=0.04)
fig2.tight_layout()
fig2.savefig('output/gaussianheatmap.png')


# Calculate the 2d cross correlation matrix
start = time()
R = norm_crosscorr2d(template, search)
end = time()
print(
    f'Time elapsed (calculate 2d cross-corr vector): {round(end - start, 2)}s'
     )

# Convert and filter cross-correlation matrix into an ordered array of
# coordinates where each dot corresponds to a single coordinate
start = time()
coords = Calibration.filter_coords(R, L)
end = time()
print(f'Time elapsed (filter coordinates): {round(end - start, 2)}s')

fig3, ax3 = plt.subplots(dpi=144, figsize=(6,6))
ax3.imshow(search, cmap='gray')
ax3.scatter(coords[:, 0], coords[:, 1], marker='s', s=40,
            facecolors='none', edgecolors='y')
ax3.set_xlabel('x (pixels)')
ax3.set_ylabel('y (pixels)')
fig3.tight_layout()
fig3.savefig('output/dotdetection.png', dpi=144)
