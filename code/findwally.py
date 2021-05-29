# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:24:13 2021

@author: Fidel
"""

# Standard libraries
from time import time
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ImageDraw import Draw

# Import module function
from crosscorrelation import norm_crosscorr2d

# Load images from data folder via Pillow
puzzle = Image.open('data/wallypuzzle.png')
target = Image.open('data/wallypuzzle_rocket.png')

# Convert images into greyscale numpy arrays
A = np.array(puzzle.convert('L'))
t = np.array(target.convert('L'))
puzzle_matrix = np.stack((A,)*3, axis=-1)
puzzle = Image.fromarray(puzzle_matrix)

# Perform 2d normalized cross-correlation and save cross-correlation matrix
print('Calculating 2d cross-correlation')
start = time()
R = norm_crosscorr2d(t, A)
end = time()
print(
    f'Time elapsed (calculate 2d cross-corr matrix): {round(end - start, 2)}s')
pd.DataFrame(R).to_csv(
    'output\wallypuzzle_crosscorr_matrix.csv', header=None, index=None)

# Locate and draw yellow rectangle around target
[y, x] = np.argwhere(R == np.max(R)).flatten()
h, w = np.shape(t)
Draw(puzzle).rectangle([x-2, y-2, x+w+2, y+h+2],
                       outline=(255, 255, 0), width=4)
puzzle.show()
puzzle.save('output/wallypuzzle_solved.png', 'png')
