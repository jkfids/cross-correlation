# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:38:16 2021

@author: Fidel
"""

# Standard libraries
from time import time
import numpy as np
from numpy import exp, array
import pandas as pd
from PIL import Image
from numba import njit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Local modules
from crosscorrelation import norm_crosscorr2d

# Global variables
a = 214.45037311374364
sigma = 6.934444436509097
shape = (17, 21)
N = shape[0]*shape[1]
threshold = 0.6


# Gaussian functions
@njit
def gaussian(x, a, sigma):
    """1D Gaussian function centred around x = 0"""
    return a*exp(-x*x/(2*sigma*sigma))


@njit
def gaussian2d(x, y, a, sigma):
    """2D Gaussian function centred around (x, y) = (0, 0)"""
    return a*exp(-(x*x+y*y)/(2*sigma*sigma))


class Calibration:
    """
    Calibration class to calibrate mapping between pixel space and real space
    """

    def __init__(self):
        self.L = 0
        self.template = array([], dtype=np.float64)
        self.variables = array([], dtype=np.float64).reshape(0, 14)
        self.labels = array([], dtype=np.float64).reshape(0, 3)
        self.coef = array([], dtype=np.float64)

    def gen_template(self, L=12):
        """
        Generate gaussian dot template array given side length L
        """
        template = np.zeros((L, L))
        template_row = np.linspace(-L/2+0.5, L/2-0.5, L)
        for i in range(L):
            # Gaussian parameters defined in Global variables
            template[i] = gaussian2d(template_row, i-(L-1)/2, a, sigma)
        self.template = template
        self.L = L
        return template

    @staticmethod
    def filter_coords(R, L):
        """
        Process the cross-correlation matrix of a calibration image to produce
        a sorted array of unique dot coordinates
        """
        N = shape[0]*shape[1]
        coords = [[]]*N
        R_coords = np.argwhere(R > threshold)
        R_coords[:, [0, 1]] = R_coords[:, [1, 0]]  # Swap columns
        R_coords = R_coords.tolist()
        # Iterate over unique coords
        for i in range(N):
            # Select first coord from R_coords
            x_i = R_coords[0][0]
            y_i = R_coords[0][1]
            x_sum = x_i
            y_sum = y_i
            R_coords_new = R_coords[1:]
            n = 1
            # Iterate through all other R_coords, adding coords close to
            # selected coord to the summation then removing them from R_coords
            for coord in R_coords[1:]:
                x = coord[0]
                y = coord[1]
                if (x_i-10 < x < x_i+10) & (y_i-10 < y < y_i+10):
                    x_sum += x
                    y_sum += y
                    n += 1
                    R_coords_new.remove(coord)
            R_coords = R_coords_new
            # Take average x_coord and y_coord of similar coords as pixel coord
            coords[i] = [round(x_sum/n + L/2), round(y_sum/n + L/2)]
        coords = array(coords)

        # Group coordinates into rows from bottom to top
        coords = coords[np.argsort(coords[:, 1])]
        # Sort elements in individual rows from left to right
        for i in range(shape[0]):
            row_i = coords[shape[1]*i:shape[1]*(i+1)]
            coords[shape[1]*i:shape[1]*(i+1)] = row_i[np.argsort(row_i[:, 0])]
        return coords

    @staticmethod
    def gen_variables(pcoords1, pcoords2):
        """Generate ordered variable array from left and right pixel coords"""
        x_l = pcoords1[:, 0]
        y_l = pcoords1[:, 1]
        x_r = pcoords2[:, 0]
        y_r = pcoords2[:, 1]
        variables = np.zeros((N, 14))
        # First order terms
        variables[:, :4] = np.hstack((pcoords1, pcoords2))
        # Second order terms
        variables[:, 4:10] = np.stack((x_l*y_l, x_l*x_r, x_l*y_r,
                                      y_l*x_r, y_l*y_r, x_r*y_r), axis=1)
        variables[:, 10:14] = variables[:, :4]**2
        return variables

    @staticmethod
    def gen_real_coords(z):
        """Generate ordered label array composed of real coords"""
        coords = np.zeros((N, 3))
        # Lower left dot is located at (-500, 0, z)
        x = np.arange(-500, 550, 50)
        y = np.arange(0, 850, 50)
        coords[:, 0] = np.tile(x, shape[0])
        coords[:, 1] = np.repeat(y, shape[1])
        coords[:, 2] = np.repeat(z, N)
        return coords

    def process_images(self, image1, image2, z):
        """
        Performs 2D cross-corr on pair of images to find and filter the pixel
        coords of calibration dots, then generate and append ordered lists of 
        real coords and pixel coord variables
        """
        # Convert PIL image into numpy array
        search1 = np.array(image1.convert('L'), dtype=np.float64)
        search2 = np.array(image2.convert('L'), dtype=np.float64)
        # Calculate cross-corr matrix between image arrays and dot template
        R1 = norm_crosscorr2d(self.template, search1)
        R2 = norm_crosscorr2d(self.template, search2)
        # Filter cross-corr matrix to obtain ordered list of pixel coordinates
        pcoords1 = self.filter_coords(R1, self.L)
        pcoords2 = self.filter_coords(R2, self.L)
        # Construct variable and label array to prepare for model fitting
        variables = self.gen_variables(pcoords1, pcoords2)
        labels = self.gen_real_coords(z)
        # Append obtained variables/labels to full array
        self.variables = np.vstack((self.variables, variables))
        self.labels = np.vstack((self.labels, labels))
        return variables, labels

    def fit_coef(self):
        """
        Obtain coeficients for 4D second order fit of pixel coords (variables)
        mapped onto real coords (labels)
        """
        model = LinearRegression()
        model.fit(self.variables, self.labels)
        intercept = model.intercept_.reshape(3, 1)
        coef = np.hstack((intercept, model.coef_))
        self.coef = coef
        
        # Print model MSE
        x = self.labels[:, 0]
        y = self.labels[:, 1]
        z = self.labels[:, 2]
        pred = model.predict(self.variables)
        MSE_x = mean_squared_error(x, pred[:, 0])
        MSE_y = mean_squared_error(y, pred[:, 1])
        MSE_z = mean_squared_error(z, pred[:, 2])
        print(f'Mean squared error (x,y,z): {MSE_x}, {MSE_y}, {MSE_z}')
        
        return coef


# Run calibration
if __name__ == '__main__':
    # Import calibration images
    left_1900 = Image.open('data/calibration/cal_image_left_1900.tiff')
    right_1900 = Image.open('data/calibration/cal_image_right_1900.tiff')
    left_1920 = Image.open('data/calibration/cal_image_left_1920.tiff')
    right_1920 = Image.open('data/calibration/cal_image_right_1920.tiff')
    left_1940 = Image.open('data/calibration/cal_image_left_1940.tiff')
    right_1940 = Image.open('data/calibration/cal_image_right_1940.tiff')
    left_1960 = Image.open('data/calibration/cal_image_left_1960.tiff')
    right_1960 = Image.open('data/calibration/cal_image_right_1960.tiff')
    left_1980 = Image.open('data/calibration/cal_image_left_1980.tiff')
    right_1980 = Image.open('data/calibration/cal_image_right_1980.tiff')
    left_2000 = Image.open('data/calibration/cal_image_left_2000.tiff')
    right_2000 = Image.open('data/calibration/cal_image_right_2000.tiff')

    # Generate template for 2d cross-correlation
    cal = Calibration()
    cal.gen_template(L=8)

    # Generate real and pixel coordinates, and prepare for fitting
    print('Processing images...')
    start = time()
    cal.process_images(left_1900, right_1900, z=1900)
    cal.process_images(left_1920, right_1920, z=1920)
    cal.process_images(left_1940, right_1940, z=1940)
    cal.process_images(left_1960, right_1960, z=1960)
    cal.process_images(left_1980, right_1980, z=1980)
    cal.process_images(left_2000, right_2000, z=2000)
    end = time()
    print(f'Time elapsed (processed 12 images): {round(end - start, 2)}s')

    # Fit linear model via linear regression and obtain coefficients
    print('Fitting coefficients...')
    cal.fit_coef()
    # Save coefficients to csv
    coefdf = pd.DataFrame(cal.coef, index=['x', 'y', 'z'])
    coefdf.to_csv('output/calibration_coef.csv') 
