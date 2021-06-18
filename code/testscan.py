# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:56:41 2021

@author: Fidel
"""

# Standard libraries
from time import time
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Local modules
from crosscorrelation import norm_crosscorr2d
from stereovision import StereoVision
from calibration import Calibration

# Import calibration data
coef = pd.read_csv('output/calibration_coef.csv', index_col=0).to_numpy()

#%%
# Test pair 1

# Obtain pixel variables for fit
start = time()
left_test1 = Image.open('data/stereo/test_left_1.tiff')
right_test1 = Image.open('data/stereo/test_right_1.tiff')
test1 = Calibration()
test1.gen_template(L=8)
test1.process_images(left_test1, right_test1, shape=(9, 13))
variables = np.ones((9*13, 15))
variables[:, 1:] = test1.variables
end = time()
print(f'Time elapsed (test pair 1): {round(end - start, 3)}s')

# Obtain real coordinates by matrix multiplication of variable and coefficient
# matrices
coords = np.matmul(variables, coef.T)
X = coords[:, 0]
Y = coords[:, 1]
Z = coords[:, 2]

# Construct 3D scatter plot
fig1 = plt.figure(dpi=144)
fig1.set_size_inches(8, 8)
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(X, Z, Y)
ax1.set_xlim3d(-500, 500)
ax1.set_ylim3d(0, 2000)
ax1.set_zlim3d(0, 800)
ax1.set_zlabel('y (mm)')
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('z (mm)')
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.set_box_aspect([1, 2, 0.8])
ax1.view_init(elev=25, azim=310)
fig1.savefig('output/testscan1.png')


#%%
# Test pair 2

left_test2 = Image.open('data/stereo/test_left_2.tiff')
right_test2 = Image.open('data/stereo/test_right_2.tiff')
test2 = StereoVision(left_test2, right_test2, calibration=coef)

start = time()
dparray2, coord_grid2 = test2.calc_dparray(64, (3,3), multipass=2)
dparray2 = test2.filter_dparray(stds=5, edge_cutoff=(2,-2,10,-8))
end = time()
print(f'Time elapsed (test pair 2): {round(end - start, 3)}s')

R = np.sqrt(dparray2[0]**2 + dparray2[1]**2)

coords = test2.calc_real_coords()
X = np.reshape(coords[:, 0], R.shape)
Y = np.reshape(coords[:, 1], R.shape)
Z = np.reshape(coords[:, 2], R.shape)
# Filter coords by threshold
Z[Z < 1915] = Z[Z > 2005] = 2000

# Construct surface plot
fig2 = plt.figure(dpi=144)
fig2.set_size_inches(8, 8)
ax2 = fig2.add_subplot(projection='3d')
surf = ax2.plot_surface(X, Z, Y, cmap='viridis')
ax2.set_xlim3d(-700, 700)
ax2.set_ylim3d(1600, 2000)
ax2.set_zlim3d(-200, 1000)
ax2.set_zlabel('y (mm)')
ax2.set_xlabel('x (mm)')
ax2.set_ylabel('z (mm)')
ax2.set_title('wsize = 64, ssize = (3,3), overlap=0, multipass=2')
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.set_box_aspect([1.4, 1, 1.2])
ax2.view_init(elev=20, azim=300)
fig2.savefig('output/testscan2.png')


#%%
# Test pair 3

left_test3 = Image.open('data/stereo/test_left_3.tiff')
right_test3 = Image.open('data/stereo/test_right_3.tiff')
test3 = StereoVision(left_test3, right_test3, calibration=coef)

start = time()
dparray3, coord_grid3 = test3.calc_dparray(64, (4,4), multipass=2)
dparray3 = test3.filter_dparray(stds=5, edge_cutoff=(4,-4,8,-8))
end = time()
print(f'Time elapsed (test pair 3): {round(end - start, 3)}s')

R = np.sqrt(dparray3[0]**2 + dparray3[1]**2)

coords = test3.calc_real_coords()
X = np.reshape(coords[:, 0], R.shape)
Y = np.reshape(coords[:, 1], R.shape)
Z = np.reshape(coords[:, 2], R.shape)
# Filter coords by threshold
Z[Z < 1915] = Z[Z > 2005] = 1960

# Construct surface plot
fig3 = plt.figure(dpi=144)
fig3.set_size_inches(8, 8)
ax3 = fig3.add_subplot(projection='3d')
surf = ax3.plot_surface(X, Z, Y, cmap='viridis')
ax3.set_xlim3d(-650, 650)
ax3.set_ylim3d(1600, 2000)
ax3.set_zlim3d(-100, 900)
ax3.set_zlabel('y (mm)')
ax3.set_xlabel('x (mm)')
ax3.set_ylabel('z (mm)')
ax3.set_title('wsize = 64, ssize = (4,4), overlap=0, multipass=2')
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.set_box_aspect([1.4, 1, 1.2])
ax3.view_init(elev=20, azim=300)
fig3.savefig('output/testscan3.png')

