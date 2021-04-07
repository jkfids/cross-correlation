# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:24:13 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np
import pandas as pd
from time import time
from PIL import Image
from PIL.ImageDraw import Draw

# Import module function
from crosscorrelation import norm_crosscorr2d
#from deprecated import norm_crosscorr2d

# Load images from data via pillow
puzzle = Image.open('data/wallypuzzle.png')
target = Image.open('data/wallypuzzle_rocket.png')

# Convert images into greyscale numpy arrays
A = np.array(puzzle.convert('RGB'))
t = np.array(target.convert('RGB'))
A = np.mean(A, axis=2)
t = np.mean(t, axis=2)
#Image.fromarray(A).show()

# Perform 2d normalized cross-correlation and save cross-correlation matrix
start = time()
R = norm_crosscorr2d(t, A)
end = time()
print(f'Time elapsed (calculate 2d cross-corr vector): {round(end - start, 2)}s')
pd.DataFrame(R).to_csv('output\wallypuzzle_crosscorr_matrix.csv', header=None, index=None)

# Locate and draw yellow rectangle around target
[y, x] = np.argwhere(R == np.max(R)).flatten() - 2
h, w = np.shape(t)
Draw(puzzle).rectangle([x, y, x+w, y+h], outline=(255,255,0), width=4)
puzzle.show()
puzzle.save('output/wallypuzzle_solved.png', 'png')
