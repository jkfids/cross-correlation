# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:24:13 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np

from crosscorrelation import *

f = [1,2,3]
g = [2,4,6]
print(crosscorr(f, g))