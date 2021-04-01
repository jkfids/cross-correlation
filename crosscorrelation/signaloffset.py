# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:28:00 2021

@author: Fidel
"""

# Import standard libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from time import time

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
sampling_freq = 44000
X = np.linspace(0, len(sensor1)/sampling_freq, len(sensor1))

fig1, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(X, sensor1, linewidth=0.2)
ax1.set_ylabel('Sensor 1')
ax2.plot(X, sensor2, linewidth=0.2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Sensor 2')
fig1.suptitle('Sensor Data')
fig1.set_size_inches(12, 4)
fig1.set_dpi(144)
fig1.savefig('output/sensordata_plot.png')

# Calculate and plot the normalized cross correlation vector
start = time()
R = norm_crosscorr(sensor1, sensor2)
#R = np.correlate(sensor1, sensor2, 'full')
end = time()
X = np.linspace(-0.5, 0.5, len(R))
print(f'Time to calculate normalized cross correlation: {round(end - start, 4)}s')

fig2, ax = plt.subplots()
ax.plot(X, R, linewidth=0.2)
ax.set_title('Sensor Cross Correlation')
ax.set_xlabel('Lag')
ax.set_ylabel('Correlation (Normalized)')
ax.xaxis.set_ticks(np.arange(-0.5, 0.6, 0.1))
fig2.set_size_inches(12, 4)
fig2.set_dpi(144)
fig2.savefig('output/sensorcorrelation_plot.png')

# Calculate time offset between two signals and hence distance between sensors
scale = 4/len(sensor1) # Sensor 1 data spans 4 seconds
offset = calc_offset(R, scale)
v = 333 # Sounds moves at 333 m/s
dist = abs(offset*v)
print(f'Time offset: Sensor 2 lags sensor 1 by {-offset} seconds')
print(f'Distance: {round(dist, 2)} metres')