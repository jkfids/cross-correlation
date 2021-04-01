# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 22:07:18 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import local modules
from crosscorrelation import *

f = np.array(list(range(1000)))
g = np.roll(f, 100)

R = norm_crosscorr(f, g)
X = np.linspace(-.5, .5, len(R))

fig, ax = plt.subplots()
ax.plot(X, R)
ax.set_title('Cross Correlation')
ax.set_xlabel('Lag')
ax.set_ylabel('Correlation (Normalized)')
ax.xaxis.set_ticks(np.arange(-0.5, 0.6, 0.1))
ax.grid(True)
fig.set_size_inches(12, 4)
fig.set_dpi(144)

#R = np.load('output/findwally_crosscorr_matrix.npy')
#pd.DataFrame(R).to_csv('output/findwally_crosscorr_matrix.csv', header=None, index=None)