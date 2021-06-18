# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:37:40 2021

@author: Fidel
"""

# Standard libraries
from time import time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.ticker as plticker

# Local modules
from stereovision import StereoVision

#%%

left_desert = Image.open('data/stereo/left_desert.png')
right_desert = Image.open('data/stereo/right_desert.png')

start = time()
desert = StereoVision(left_desert, right_desert, resize=0.5)
dparray, _ = desert.calc_dparray(16, (7,3))
end = time()
print(f'Time elapsed (axis comparison): {round(end - start, 3)}s')

dparray = desert.filter_dparray(passes=0, edge_cutoff=(0,-1,0,-2))
X, Y = dparray
X[X<10] = np.mean(X)
Y[Y>19] = Y[Y<0] = np.mean(Y)
R = np.sqrt(X*X + Y*Y)

fig1, axes1 = plt.subplots(1, 3, dpi=144, figsize=(8,6))
im = axes1[0].imshow(X, vmin=0, vmax=50)
im = axes1[1].imshow(Y, vmin=0, vmax=50)
im = axes1[2].imshow(R, vmin=0, vmax=50)
axes1[0].set_title('Δx')
axes1[1].set_title('Δy')
axes1[2].set_title('Δr')
fig1.tight_layout(rect=[0, 0, 0.95, 1])
fig1.colorbar(im, ax=axes1.ravel().tolist(), fraction=0.018)
fig1.savefig('output/desert_dparray')

#%%
left_desert = Image.open('data/stereo/left_desert.png')
right_desert = Image.open('data/stereo/right_desert.png')

desert1 = StereoVision(left_desert, right_desert, resize=0.5)
desert2 = StereoVision(left_desert, right_desert, resize=0.5)
desert3 = StereoVision(left_desert, right_desert, resize=0.5)
time1 = time()
dparray1, _ = desert1.calc_dparray(16, (7,3), overlap=.25)
dparray1 = desert1.filter_dparray(passes=2, edge_cutoff=(0,-1,0,-3))
time2 = time()
dparray2, _ = desert2.calc_dparray(16, (7,3), overlap=.5)
dparray2 = desert2.filter_dparray(passes=2, edge_cutoff=(0,-2,0,-4))
time3 = time()
dparray3, _ = desert3.calc_dparray(16, (7,3), overlap=.75)
dparray3 = desert3.filter_dparray(passes=2, edge_cutoff=(0,-4,0,-7))
time4 = time()
print(f'Time elapsed (overlap = 0.25): {round(time2 - time1, 3)}s')
print(f'Time elapsed (overlap = 0.5): {round(time3 - time2, 3)}s')
print(f'Time elapsed (overlap = 0.75): {round(time4 - time3, 3)}s')

X1, Y1 = dparray1
X2, Y2 = dparray2
X3, Y3 = dparray3
R1 = np.sqrt(X1*X1 + Y1*Y1)
R2 = np.sqrt(X2*X2 + Y2*Y2)
R3 = np.sqrt(X3*X3 + Y3*Y3)

fig2, axes2 = plt.subplots(1, 3, dpi=144, figsize=(8,6))
im = axes2[0].imshow(R1, vmin=0, vmax=50)
im = axes2[1].imshow(R2, vmin=0, vmax=50)
im = axes2[2].imshow(R3, vmin=0, vmax=50)
axes2[0].set_title('overlap = 0.25', fontsize=9)
axes2[1].set_title('overlap = 0.5', fontsize=9)
axes2[2].set_title('overlap = 0.75', fontsize=9)
fig2.tight_layout(rect=[0, 0, 0.95, 1])
fig2.colorbar(im, ax=axes2.ravel().tolist(), fraction=0.018)
fig2.savefig('output/overlap_dparray')


#%%
left_desert = Image.open('data/stereo/left_desert.png')
right_desert = Image.open('data/stereo/right_desert.png')

desert1 = StereoVision(left_desert, right_desert, resize=0.5)
desert2 = StereoVision(left_desert, right_desert, resize=0.5)
desert3 = StereoVision(left_desert, right_desert, resize=0.5)
time1 = time()
dparray1, _ = desert1.calc_dparray(16, (12,8), multipass=2)
dparray1 = desert1.filter_dparray(edge_cutoff=(0,-4,0,-6))
time2 = time()
dparray2, _ = desert2.calc_dparray(32, (6,4), multipass=3)
dparray2 = desert2.filter_dparray(edge_cutoff=(0,-4,0,-4))
time3 = time()
dparray3, _ = desert3.calc_dparray(64, (3,2), multipass=4)
dparray3 = desert3.filter_dparray(edge_cutoff=(0,-4,0,None))
time4 = time()
print(f'Time elapsed (multipass = 2): {round(time2 - time1, 3)}s')
print(f'Time elapsed (multipass = 3): {round(time3 - time2, 3)}s')
print(f'Time elapsed (multipass = 4): {round(time4 - time3, 3)}s')

X1, Y1 = dparray1
X2, Y2 = dparray2
X3, Y3 = dparray3
R1 = np.sqrt(X1*X1 + Y1*Y1)
R2 = np.sqrt(X2*X2 + Y2*Y2)
R3 = np.sqrt(X3*X3 + Y3*Y3)
end = time()

fig3, axes3 = plt.subplots(1, 3, dpi=144, figsize=(8,6))
im = axes3[0].imshow(R1, vmin=0, vmax=50)
im = axes3[1].imshow(R2, vmin=0, vmax=50)
im = axes3[2].imshow(R3, vmin=0, vmax=50)
axes3[0].set_title('multipass level = 2', fontsize=8)
axes3[1].set_title('multipass level = 3', fontsize=8)
axes3[2].set_title('multipass level = 4', fontsize=8)
fig3.tight_layout(rect=[0, 0, 0.95, 1])
fig3.colorbar(im, ax=axes3.ravel().tolist(), fraction=0.018)
fig3.savefig('output/multipass_dparray')

#%%

left_portal = Image.open('data/stereo/left_portal.tiff')
right_portal = Image.open('data/stereo/right_portal.tiff')

portal = StereoVision(left_portal, right_portal)
start = time()
dparray, _ = portal.calc_dparray(32, (3,1), overlap=0.75, multipass=1)
dparray = portal.filter_dparray(stds=4, edge_cutoff=(2,None,0,None))
end = time()
print(f'Time elapsed (portal images): {round(end - start, 3)}s')

X, Y = dparray
R = np.sqrt(X*X + Y*Y)

fig4, ax4 = plt.subplots(dpi=144, figsize=(8,6))
im = ax4.imshow(R)
ax4.set_title('wsize = 32, ssize = (3,1), overlap = 0.75, multipass = 1')
fig4.colorbar(im, ax=ax4, fraction=0.03, pad=0.04)
fig4.savefig('output/portal_dparray')

#%%

left_cone = Image.open('data/stereo/left_cone.tiff')
right_cone = Image.open('data/stereo/right_cone.tiff')

cone = StereoVision(left_cone, right_cone)
start = time()
dparray, _ = cone.calc_dparray(64, (2,2), multipass=4)
dparray = cone.filter_dparray(stds=2)
end = time()
print(f'Time elapsed (cone images): {round(end - start, 3)}s')

X, Y = dparray
R = np.sqrt(X*X + Y*Y)

fig5, ax5 = plt.subplots(dpi=144, figsize=(6,8))
im = ax5.imshow(R)
ax5.set_title('wsize = 64, ssize = (2,2), overlap = 0, multipass = 4', fontsize=12)
fig5.colorbar(im, ax=ax5, fraction=0.08)
fig5.savefig('output/cone_dparray')

