# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:14:46 2021

@author: Fidel
"""

# Standard libraries
import numpy as np
from numpy import exp, array
from numba import njit
from time import time

# Local modules
from crosscorrelation import norm_crosscorr2d

# Global variables
a = 214.45037311374364
sigma = 6.934444436509097
shape = (17, 21)
N = shape[0]*shape[1]
threshold = 0.6

@njit
def gaussian(x, a, sigma):
    """1D Gaussian function centred around x = 0"""
    return a*exp(-x*x/(2*sigma*sigma))

@njit
def gaussian2d(x, y, a, sigma):
    """2D Gaussian function centred around (x, y) = (0, 0)"""
    return a*exp(-(x*x+y*y)/(2*sigma*sigma))

def filter_coords(R, L=12):
    """
    Process the cross-correlation matrix of a calibration image to produce a 
    sorted array of unique dot coordinates
    """
    N = shape[0]*shape[1]
    coords = [[]]*N
    R_coords = np.argwhere(R > threshold)
    R_coords[:,[0, 1]] = R_coords[:,[1, 0]] # Swap columns
    R_coords = R_coords.tolist()
    # Iterate over unique coordinates
    for i in range(N):
        # Select first coordinate from R_coords
        x_i = R_coords[0][0]
        y_i = R_coords[0][1]
        x_sum = x_i
        y_sum = y_i
        R_coords_new = R_coords[1:]
        n = 1
        # Iterate through remaining R_coords, adding coords similar to
        # selected coord to summation then removing them from R_coords
        for coord in R_coords[1:]:
            x = coord[0]
            y = coord[1]
            if (x_i-10 < x < x_i+10) & (y_i-10 < y < y_i+10):
                x_sum += x
                y_sum += y
                n += 1
                R_coords_new.remove(coord)
        R_coords = R_coords_new
        # Take average x_coord and y_coord of similar coords as final pixel coord
        coords[i] = [round(x_sum/n + L/2), round(y_sum/n + L/2)]
    coords = array(coords)
    
    # Group coordinates into rows from bottom to top
    coords = coords[np.argsort(coords[:,1])]
    # Sort row elements from left to right
    for i in range(shape[0]):
        row_i = coords[shape[1]*i:shape[1]*(i+1)]
        coords[shape[1]*i:shape[1]*(i+1)] = row_i[np.argsort(row_i[:,0])]
    return coords

class Calibration:
    """
    """
    def __init__(self):
        self.L = 0
        self.template = array([], dtype=np.float64)
        self.variables = array([], dtype=np.float64).reshape(0,14)
        self.labels = array([], dtype=np.float64).reshape(0,3)
        
    def gen_template(self, L=12):
        """
        """
        template = np.zeros((L, L))
        template_row = np.linspace(-L/2+0.5, L/2-0.5, L)
        for i in range(L):
            template[i] = gaussian2d(template_row, i-(L-1)/2, a, sigma)
        self.template = template
        self.L = L
        return template
    
    @staticmethod
    def gen_real_coords(z):
        """"""
        coords = np.zeros((N, 3))
        x = np.arange(-500, 550, 50)
        y = np.arange(0, 850, 50)
        coords[:,0] = np.tile(x, shape[0])
        coords[:,1] = np.repeat(y, shape[1])
        coords[:,2] = np.repeat(z, N)
        return coords
    
    @staticmethod
    def gen_variables(left_pixel_coords, right_pixel_coords):
        """"""
        x_l = left_pixel_coords[:, 0]
        y_l = left_pixel_coords[:, 1]
        x_r = right_pixel_coords[:, 0]
        y_r = right_pixel_coords[:, 1]
        variables = np.zeros((N, 14))
        variables[:, :4] = np.hstack((left_pixel_coords, right_pixel_coords))
        variables[:, 4:10] = np.stack((x_l*y_l, x_l*x_r, x_l*y_r, \
                                      y_l*x_r, y_l*y_r, x_r*y_r), axis=1)
        variables[:, 10:14] = variables[:, :4]**2
        return variables
    
    def process_images(self, left_image, right_image, z):
        """
        """
        # Convert PIL image into numpy array
        left_search = np.array(left_image.convert('L'))
        right_search = np.array(right_image.convert('L'))
        # Calculate cross-corr matrix between image arrays and dot template
        left_R = norm_crosscorr2d(self.template, left_search)
        right_R = norm_crosscorr2d(self.template, right_search)
        # Filter cross-corr matrix to obtain ordered list of pixel coordinates
        left_pixel_coords = filter_coords(left_R, self.L)
        right_pixel_coords = filter_coords(right_R, self.L)
        #
        variables = self.gen_variables(left_pixel_coords, right_pixel_coords)
        labels = self.gen_real_coords(z)
        self.variables = np.vstack((self.variables, variables))
        self.labels = np.vstack((self.labels, labels))
        return variables, labels