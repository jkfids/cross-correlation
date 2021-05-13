# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:38:16 2021

@author: Fidel
"""

# Standard libraries
import numpy as np
import pandas as pd
from PIL import Image
from time import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Local modules
from stereovision import Calibration

# Import calibration images
left_1900 = Image.open('data/cal_image_left_1900.tiff')
right_1900 = Image.open('data/cal_image_right_1900.tiff')
left_1920 = Image.open('data/cal_image_left_1920.tiff')
right_1920 = Image.open('data/cal_image_right_1920.tiff')
left_1940 = Image.open('data/cal_image_left_1940.tiff')
right_1940 = Image.open('data/cal_image_right_1940.tiff')
left_1960 = Image.open('data/cal_image_left_1960.tiff')
right_1960 = Image.open('data/cal_image_right_1960.tiff')
left_1980 = Image.open('data/cal_image_left_1980.tiff')
right_1980 = Image.open('data/cal_image_right_1980.tiff')
left_2000 = Image.open('data/cal_image_left_2000.tiff')
right_2000 = Image.open('data/cal_image_right_2000.tiff')

# Generate template for 2d cross-correlation
cal = Calibration()
cal.gen_template(L=12)

# Generate pixel coordinates, real coordinates, and prepare for fitting 
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

x = cal.labels[:,0]
y = cal.labels[:,1]
z = cal.labels[:,2]

# Fit linear model via linear regression and obtain coefficients
print('Fitting coefficients...')
model = LinearRegression()
model.fit(cal.variables, cal.labels)

intercept = model.intercept_.reshape(3,1)
coef = np.hstack((intercept, model.coef_))
pd.DataFrame(coef).to_csv('output/calibration_coef.csv', header=None, index=None)

# Print mean squared error of model fit
pred = model.predict(cal.variables)
MSE_x = mean_squared_error(x, pred[:,0])
MSE_y = mean_squared_error(y, pred[:,1])
MSE_z = mean_squared_error(z, pred[:,2])
print(f'Mean squared error (x,y,z): {MSE_x}, {MSE_y}, {MSE_z}')
