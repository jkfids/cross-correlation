# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:28:00 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import local modules
from crosscorrelation import *

# Create output directory if it does not exist
from pathlib import Path
Path('output').mkdir(parents=True, exist_ok=True)

# Import sensor data via pandas
df1 = pd.read_csv('data\sensor1_data.txt', delimiter = "\t")
df2 = pd.read_csv('data\sensor2_data.txt', delimiter = "\t")
sensor1 = df1.to_numpy().flatten()
sensor2 = df2.to_numpy().flatten()

# Plot sensor data
X = np.linspace(0, len(sensor1)/44000, len(sensor1))

fig1, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(X, sensor1, linewidth=0.1)
ax1.set_ylabel('Sensor 1')
ax2.plot(X, sensor2, linewidth=0.1)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Sensor 2')
fig1.suptitle('Sensor Data')
fig1.savefig('output/sensordata_plot.png')

# Calculate and plot the normalized cross correlation vector
R = norm_crosscorr(sensor1, sensor2)
X = np.linspace(0, 1, len(R))

fig2, ax = plt.subplots()
ax.plot(X, R, linewidth=0.2)
ax.set_title('Sensor Cross Correlation')
ax.set_xlabel('Lag')
ax.set_ylabel('Correlation (Normalized)')
fig2.savefig('output/sensorcorrelation_plot.png')
