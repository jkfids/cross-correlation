# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:24:13 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np
from PIL import Image

from crosscorrelation import *

# Load images from data and convert into greyscale numpy arrays

puzzle = Image.open('data\wallypuzzle.png')
rocket = Image.open('data\wallypuzzle_rocket.png')
#puzzle.show()

A = np.array(puzzle.convert('RGB'))
A = np.mean(A, axis=2)
t = np.array(rocket.convert('RGB'))
t = np.mean(t, axis=2)

#Image.fromarray(A).show()


