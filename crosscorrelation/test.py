# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 22:07:18 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np
from matplotlib import pyplot as plt

# Import local modules
from crosscorrelation import norm_crosscorr

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

arg_max = np.argmax(R)/(len(R)-1)-0.5
print(arg_max)
print(len(f)-1-np.argmax(R))
