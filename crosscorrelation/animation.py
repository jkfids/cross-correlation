# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:26:53 2021

@author: Fidel
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from crosscorrelation import *

# Set parameters
length = 100

fig, (ax1, ax2, ax3) = plt.subplots(3,1, dpi=144)
line1, = ax1.plot([], [], lw=2, color='tab:blue')
line2, = ax2.plot([], [], lw=2, color='tab:red')
line3, = ax3.plot([], [], lw=2, color='purple')
ax1.set_xlim([0, 1])
ax1.set_ylim([-1.5, 1.5])
ax2.set_xlim([0, 1])
ax2.set_ylim([-1.5, 1.5])
ax3.set_xlim([-.5, .5])
ax3.set_ylim([-1.2, 1.2])
ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax1.set_yticks([])
ax2.set_yticks([])
ax3.set_yticks([])

x = np.linspace(0, 1, length)
y = np.sin(5*2*np.pi*x)
x = np.linspace(0, 1, round(1.5*length))
y = np.pad(y, (round(length/4), round(length/4)))
lenx = len(x)
leny = len(y)
y_slide = np.pad(y, (lenx-1, lenx-1))
leny_slide = len(y_slide)
x_R = np.linspace(-.5, .5, 2*lenx-1)
R = norm_crosscorr(y, y)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3,
def animate(i):
    y_subset = y_slide[leny_slide-i-1-lenx:leny_slide-i-1]
    line1.set_data(x, y)
    line2.set_data(x, y_subset)
    line3.set_data(x_R[1:i+1], R[1:i+1])
    print(i)
    return line1, line2, line3,


anim = FuncAnimation(fig, animate, init_func=init, frames=len(R)-1, interval=50, blit=True)
anim.save('output/crosscorrelation.gif', writer='ffmpeg')