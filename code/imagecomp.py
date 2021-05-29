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

# Local modules
from stereovision import StereoVision

image1 = Image.open('data/stereo/left_desert.png')
image2 = Image.open('data/stereo/right_desert.png')

w, h = image1.size
image1 = image1.resize((round(w/2), round(h/2)))
image2 = image2.resize((round(w/2), round(h/2)))

#image1 = Image.open('data/stereo/left_portal.tiff')
#image2 = Image.open('data/stereo/right_portal.tiff')

#image1 = Image.open('data/test_left_1.tiff')
#image2 = Image.open('data/test_right_1.tiff')

#dpx, dpy = calc_shift(image1, image2, 64, 175, 215, (3,3))
#print(dpx, dpy)

test = StereoVision(64, overlap=0, ssize=(3,3), multipass_level=4)
start = time()
dparray = test.calc_dparray(image1, image2)
end = time()
print(f'Time elapsed: {round(end - start, 3)}s')

#dparray = dparray[:,:-1,:-1]
x = dparray[0]**2
y = dparray[1]**2
dprms = np.sqrt(x+y)
plt.imshow(dprms)
plt.colorbar()
