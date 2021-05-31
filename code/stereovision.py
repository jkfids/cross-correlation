# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:14:46 2021

@author: Fidel
"""

# Standard libraries
import numpy as np
from math import floor, ceil
import matplotlib.pyplot as plt

# Local modules
from crosscorrelation import norm_crosscorr2d


def calc_shift(image1, image2, wsize, xgrid, ygrid, ssize=(3, 3)):
    """"""
    # Convert PIL images into numpy arrays
    imarray1 = np.array(image1.convert('L'))
    imarray2 = np.array(image2.convert('L'))
    # Pad image2 array for edge cases
    xpad = ceil(wsize*(ssize[0]-1)/2)
    ypad = ceil(wsize*(ssize[1]-1)/2)
    imarray2 = np.pad(imarray2, ((ypad, ypad), (xpad, xpad)))

    h1 = wsize/2
    h2x = round(wsize*ssize[0]/2)
    h2y = round(wsize*ssize[1]/2)
    t = imarray1[ygrid - floor(h1):ygrid + ceil(h1),
                 xgrid - floor(h1):xgrid + ceil(h1)]
    A = imarray2[ygrid + ypad - h2y:ygrid + ypad + h2y,
                 xgrid + xpad - h2x:xgrid + xpad + h2x]
    R = norm_crosscorr2d(t, A)

    y, x = np.argwhere(R == np.max(R)).flatten()
    dpx = round(x - wsize*(ssize[0]-1)/2)
    dpy = round(y - wsize*(ssize[1]-1)/2)
    return [dpx, dpy]


class StereoVision:
    """"""

    def __init__(self, wsize, ssize=(3, 3), overlap=0, multipass_level=1):
        self.wsize = wsize
        self.ssize = ssize
        self.overlap = overlap
        self.multipass_level = multipass_level

        self.dparray = np.array([])

    def calc_shift(self, imarray1, imarray2, xgrid, ygrid, xguess=0, yguess=0):
        """"""
        # Pad imarray2 for edge cases
        xpad = ceil(self.wsize*(self.ssize[0]-1)/2)
        ypad = ceil(self.wsize*(self.ssize[1]-1)/2)
        imarray2 = np.pad(imarray2, ((ypad, ypad), (xpad, xpad)))

        h1 = self.wsize/2
        h2x = round(self.wsize*self.ssize[0]/2)
        h2y = round(self.wsize*self.ssize[1]/2)
        t = imarray1[ygrid - floor(h1):ygrid + ceil(h1),
                     xgrid - floor(h1):xgrid + ceil(h1)]
        A = imarray2[ygrid + yguess + ypad - h2y:ygrid + yguess + ypad + h2y,
                     xgrid + xguess + xpad - h2x:xgrid + xguess + xpad + h2x]
        try:
            R = norm_crosscorr2d(t, A)
            y, x = np.argwhere(R == np.max(R)).flatten()
            dpx = round(x - self.wsize*(self.ssize[0]-1)/2)
            dpy = round(y - self.wsize*(self.ssize[1]-1)/2)
            return [dpx + xguess, dpy + yguess]
        except:
            return [0, 0]

    def calc_dparray(self, image1, image2):
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
        # Initialise range of centre x, y-coordinates (considering overlap)
        # and dparray
        xcoords = np.arange(round(self.wsize/2), w-round(self.wsize/2)+1,
                            round(self.wsize*(1-self.overlap)))
        ycoords = np.arange(round(self.wsize/2), h-round(self.wsize/2)+1,
                            round(self.wsize*(1-self.overlap)))
        dparray = np.zeros((2, ycoords.size, xcoords.size), dtype=np.int32)
        # Repeat dparray calculation    
        for level in range(self.multipass_level):
            for i, ycoord in enumerate(ycoords):
                for j, xcoord in enumerate(xcoords):
                    dparray[:, i, j] = self.calc_shift(
                        imarray1, imarray2, xcoord, ycoord, 
                        dparray[0, i, j], dparray[1, i, j]
                        )
            if level == self.multipass_level - 1:
                self.dparray = dparray
                return dparray 
            self.wsize = round(self.wsize/2)
            xcoords = np.dstack((xcoords-round(self.wsize/2), 
                                 xcoords+round(self.wsize/2))).flatten()
            ycoords = np.dstack((ycoords-round(self.wsize/2), 
                                 ycoords+round(self.wsize/2))).flatten()
            dparray = np.kron(dparray, np.ones((2,2), dtype=np.int32))
    
        
