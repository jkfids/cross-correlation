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

R = norm_crosscorr(f, f)
X = np.linspace(-1, 1, len(R))

fig, ax = plt.subplots()
ax.plot(X, R)
ax.set_title('Cross Correlation')
ax.set_xlabel('Lag')
ax.set_ylabel('Correlation (Normalized)')
fig.set_size_inches(12, 4)
fig.set_dpi(144)

print(np.argmax(R))