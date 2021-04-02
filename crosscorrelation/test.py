# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 22:07:18 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from time import time

# Import local modules
from crosscorrelation import *
import deprecated

#%%
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

#%%
A = np.random.rand(100,100)
t = np.random.rand(10,10)
f = np.array(list(range(10000)))
#R = norm_crosscorr(f,f)
start = time()
#R = norm_crosscorr(f,f)
A = np.random.rand(1000,1000)
t = np.random.rand(10,10)
R = norm_crosscorr2d(t,A)
end = time()
print(end-start)
