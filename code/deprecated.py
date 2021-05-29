# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:50:42 2021

@author: Fidel
"""
# Import
import numpy as np
from PIL import Image

image = Image.open('data/desert.png')
#image.show()
w, h = image.size
left = image.crop((0, 0, round(w/2-1), h))
right = image.crop((round(w/2+1), 0, w, h))
print(left.size, right.size)
right.show()
left.save('data/stereo/left_desert.png', 'png')
right.save('data/stereo/right_desert.png', 'png')

# Deprecated functions
def crosscorr(f, g):
    """
    Takes two vectors of the same size, subtracts the vector elements by their
    respective means, and passes one over the other to construct a 
    cross-correlation vector
    """
    N = len(f)
    r = np.array([], dtype=np.single)
    r1 = np.array([], dtype=np.single)
    r2 = np.array([], dtype=np.single)
    f = f - np.mean(f)
    g = g - np.mean(g)
    
    for i in range(N-1):
        r1i = np.dot(f[N-i-1:N], g[0:i+1])
        r2i = np.dot(f[0:N-i-1], g[i+1:N])
        r1 = np.append(r1, r1i)
        r2 = np.append(r2, r2i)
        
    r = np.append(r, r1)
    r = np.append(r, np.dot(f, g))
    r = np.append(r, r2)
    return r/N

def process_images_(self, image1, image2):
    """"""
    # Convert PIL image into numpy array
    imarray1 = np.array(image1.convert('L'))
    imarray2 = np.array(image2.convert('L'))
    # Pad image arrays for evenly sized windows
    h, w = imarray1.shape
    xpad = self.wsize - w % self.wsize
    ypad = self.wsize - h % self.wsize
    imarray1 = np.pad(imarray1, ((0, ypad), (0, xpad)))
    imarray2 = np.pad(imarray2, ((0, ypad), (0, xpad)))
    # Calculate the number of windows along an axis given overlap
    nx = round((w/self.wsize)/(1-self.overlap)) + 1
    ny = round((h/self.wsize)/(1-self.overlap)) + 1
    # Construct 3D array of pixel differences
    dparray = np.zeros((2, ny, nx))
    for i in range(ny):
        ygrid = round((i + 0.5)*(1-self.overlap)*self.wsize)
        for j in range(nx):
            xgrid = round((j + 0.5)*(1-self.overlap)*self.wsize)
            dparray[:, i, j] = self.calc_shift(
                imarray1, imarray2, xgrid, ygrid)
    return dparray
