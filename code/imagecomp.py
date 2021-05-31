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

"""
image = Image.open('data/newyork.jpg')
w, h = image.size
image2 = image.crop((0, 0, round(w/2), h))
image1 = image.crop((round(w/2), 0, w, h))
"""

image1 = Image.open('data/stereo/left_desert.png')
image2 = Image.open('data/stereo/right_desert.png')

#image1 = Image.open('data/stereo/left_cone.tiff')
#image2 = Image.open('data/stereo/right_cone.tiff')

#image1 = Image.open('data/stereo/test_left_2.tiff')
#image2 = Image.open('data/stereo/test_right_2.tiff')

#dpx, dpy = calc_shift(image1, image2, 64, 175, 215, (3,3))
#print(dpx, dpy)

start = time()
test = StereoVision(image1, image2, resize=0.5)
dparray, _ = test.calc_dparray(64, (3,2), overlap=0, multipass_level=4)
end = time()
print(f'Time elapsed: {round(end - start, 3)}s')

dparray[0][dparray[0]<0] = np.mean(dparray[0])

#dparray = dparray[:,:-1,:-1]
x = dparray[0]**2
y = dparray[1]**2
dprms = np.sqrt(x+y)
plt.imshow(dprms)
plt.imshow(dparray[0])
plt.colorbar()
