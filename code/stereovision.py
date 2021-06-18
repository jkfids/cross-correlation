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
    """
    Stereo vision class for depth extraction from stereo images
    """

    def __init__(self, image1, image2, resize=1, calibration=None):
        w, h = image1.size
        self.image1 = image1.resize((round(resize*w), round(resize*h)))
        self.image2 = image2.resize((round(resize*w), round(resize*h)))
        self.calibration = calibration

        self.wsize = 0
        self.ssize = ()
        self.overlap = 0
        self.multipass = 0
        self.dparray = np.array([])
        
        self.real_coords = np.array([])

    def calc_shift(self, imarray1, imarray2, x, y, x0=0, y0=0):
        """
        Calculate the pixel shift between a subset of the stereo image pair
        """
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

    def calc_dparray(self, wsize, ssize=(3, 3), overlap=0, multipass=1):
        """
        Split up the stereo images into even sized windows, calculating the
        pixel shift for each. Contains optimisation methods including variable
        overlap, variable window size, and multipass cross-correlation.
        """
        self.wsize = wsize
        self.ssize = ssize
        self.overlap = overlap
        self.multipass = multipass
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
        for level in range(multipass):
            for i, y in enumerate(ycoords):
                for j, x in enumerate(xcoords):
                    dparray[:, i, j] = self.calc_shift(
                        imarray1, imarray2, x, y, 
                        dparray[0, i, j], dparray[1, i, j]
                        )
            if level == multipass - 1:
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
            
    def filter_dparray(self, stds=3, passes=1, edge_cutoff=(0,None,0,None)):
        """
        Remove spurious vectors by setting elements outside a multiple of the
        standard deviation to the mean. Also has the option to crop the
        dparray and coord grid
        """
        # Edge cutoff
        a, b, c, d = edge_cutoff
        self.dparray = self.dparray[:, a:b, c:d]
        self.coord_grid[0] = self.coord_grid[0][a:b, c:d]
        self.coord_grid[1] = self.coord_grid[1][a:b, c:d]
        
        X, Y = self.dparray
        # Repeat for number of passes
        for i in range(passes):
            mean_X = np.mean(X)
            mean_Y = np.mean(Y)
            std_X = np.std(X)
            std_Y = np.std(Y)
            
            X[X < mean_X-stds*std_X] = mean_X
            X[X > mean_X+stds*std_X] = mean_X
            Y[Y < mean_Y-stds*std_Y] = mean_Y
            Y[Y > mean_Y+stds*std_Y] = mean_Y
        
        self.dparray = np.stack((X, Y))
        return self.dparray
    
    def calc_real_coords(self):
        """Map the real coordinates from coord_grid using calibration data"""
        
        x_l = self.coord_grid[0].flatten()
        y_l = self.coord_grid[1].flatten()
        x_r = x_l + self.dparray[0].flatten()
        y_r = y_l + self.dparray[1].flatten()
        
        variables = np.ones((x_l.size, 15))
        variables[:, 1:5] = np.stack((x_l, y_l, x_r, y_r), axis=1)
        variables[:, 5:11] = np.stack((x_l*y_l, x_l*x_r, x_l*y_r,
                                      y_l*x_r, y_l*y_r, x_r*y_r), axis=1)
        variables[:, 11:15] = variables[:, 1:5]**2
        
        self.real_coords = np.matmul(variables, self.calibration.T)
        return self.real_coords
