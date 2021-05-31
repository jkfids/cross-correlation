# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:14:46 2021

@author: Fidel
"""

# Standard libraries
import numpy as np
from math import floor, ceil

# Local modules
from crosscorrelation import norm_crosscorr2d


class StereoVision:
    """"""

    def __init__(self, image1, image2, resize=1, calibration=None):
        w, h = image1.size
        self.image1 = image1.resize((round(resize*w), round(resize*h)))
        self.image2 = image2.resize((round(resize*w), round(resize*h)))
        self.calibration = calibration

        self.wsize = 0
        self.ssize = ()
        self.overlap = 0
        self.multipass_level = 0
        self.dparray = np.array([])

    def calc_shift(self, imarray1, imarray2, x, y, x0=0, y0=0):
        """"""
        # Pad imarray2 for edge cases
        xpad = ceil(self.wsize*(self.ssize[0]-1)/2)
        ypad = ceil(self.wsize*(self.ssize[1]-1)/2)
        imarray2 = np.pad(imarray2, ((ypad, ypad), (xpad, xpad)))

        h1 = self.wsize/2
        h2x = round(self.wsize*self.ssize[0]/2)
        h2y = round(self.wsize*self.ssize[1]/2)
        t = imarray1[y - floor(h1):y + ceil(h1),
                     x - floor(h1):x + ceil(h1)]
        A = imarray2[y + y0 + ypad - h2y:y + y0 + ypad + h2y,
                     x + x0 + xpad - h2x:x + x0 + xpad + h2x]
        try:
            R = norm_crosscorr2d(t, A)
            y, x = np.argwhere(R == np.max(R)).flatten()
            dpx = round(x - self.wsize*(self.ssize[0]-1)/2)
            dpy = round(y - self.wsize*(self.ssize[1]-1)/2)
            return [dpx + x0, dpy + y0]
        except:
            return [0, 0]

    def calc_dparray(self, wsize, ssize=(3, 3), overlap=0, multipass_level=1):
        """"""
        self.wsize = wsize
        self.ssize = ssize
        self.overlap = overlap
        self.multipass_level = multipass_level
        # Convert PIL image into numpy array
        imarray1 = np.array(self.image1.convert('L'))
        imarray2 = np.array(self.image2.convert('L'))
        # Pad image arrays for evenly sized windows
        h, w = imarray1.shape
        xpad = wsize - w % wsize
        ypad = wsize - h % wsize
        imarray1 = np.pad(imarray1, ((0, ypad), (0, xpad)))
        imarray2 = np.pad(imarray2, ((0, ypad), (0, xpad)))
        # Initialise range of centre x, y-coordinates (considering overlap)
        # coord_grid and dparray
        xcoords = np.arange(round(wsize/2), w-round(wsize/2)+1, 
                            round(wsize*(1-overlap)))
        ycoords = np.arange(round(wsize/2), h-round(wsize/2)+1, 
                            round(wsize*(1-overlap)))
        coord_grid = np.meshgrid(xcoords, ycoords)
        dparray = np.zeros((2, ycoords.size, xcoords.size), dtype=np.int32)
        # Repeat dparray calculation for number of passes
        for level in range(multipass_level):
            for i, y in enumerate(ycoords):
                for j, x in enumerate(xcoords):
                    dparray[:, i, j] = self.calc_shift(
                        imarray1, imarray2, x, y, 
                        dparray[0, i, j], dparray[1, i, j]
                        )
            if level == multipass_level - 1:
                self.coord_grid = coord_grid
                self.dparray = dparray
                return dparray, coord_grid
            # Update wsize, coord_grid and dparray for next pass
            self.wsize = wsize = round(wsize/2) # Halve wsize
            xcoords = np.dstack((xcoords-round(wsize/2),
                                 xcoords+round(wsize/2))).flatten()
            ycoords = np.dstack((ycoords-round(wsize/2),
                                 ycoords+round(wsize/2))).flatten()
            coord_grid = np.meshgrid(xcoords, ycoords)
            dparray = np.kron(dparray, np.ones((2, 2), dtype=np.int32))
