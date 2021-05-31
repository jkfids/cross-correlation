# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:28:00 2021

@author: Fidel
"""

# Standard libraries
from time import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Local modules
from crosscorrelation import norm_crosscorr, spectral_crosscorr, calc_offset

# Import sensor data via pandas
df1 = pd.read_csv('data/sensor1_data.txt', delimiter='\t', header=None)
df2 = pd.read_csv('data/sensor2_data.txt', delimiter='\t', header=None)
sensor1 = df1.to_numpy().flatten()
sensor2 = df2.to_numpy().flatten()

# Plot sensor data
sampling_freq = 44000
length = len(sensor1)
X1 = np.linspace(0, length/sampling_freq, len(sensor1))

fig1, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(X1, sensor1, linewidth=0.2)
ax1.set_ylabel('Sensor 1')
ax2.plot(X1, sensor2, linewidth=0.2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Sensor 2')
fig1.suptitle('Sensor Data')
fig1.set_size_inches(12, 4)
fig1.set_dpi(144)
fig1.savefig('output/sensordata_plot.png')

# Calculate and plot the normalized cross correlation vector
start = time()
R1 = norm_crosscorr(sensor1, sensor2)
end = time()
X2 = np.linspace(-0.5, 0.5, len(R1))
print(f'Time elapsed (calculate crosscorr vector): {round(end - start, 2)}s')
start = time()
R2 = np.correlate(sensor1, sensor2, 'full')
end = time()
print(f'Time elapsed (calculate numpy crosscorr): {round(end - start, 2)}s')
start = time()
R3 = spectral_crosscorr(sensor1, sensor2)
end = time()
X3 = np.linspace(-0.5, 0.5, len(R3))
print(f'Time elapsed (calculate spectral crosscorr): {round(end - start, 2)}s')

fig2, ax = plt.subplots()
ax.plot(X2, R1, linewidth=0.2)
ax.set_title('Sensor Cross Correlation')
ax.set_xlabel('Lag')
ax.set_ylabel('Correlation (Normalized)')
ax.xaxis.set_ticks(np.arange(-0.5, 0.6, 0.1))
fig2.set_size_inches(12, 4)
fig2.set_dpi(144)
fig2.savefig('output/sensorcorrelation_plot.png')

# Calculate time offset between two signals and hence distance between sensors
scale = 1/44000  # Sampling rate of 44 kHz
offset = calc_offset(R1, scale, mode='spatial')
#offset = calc_offset(R3, scale, mode='spectral')
v = 333  # Sounds moves at 333 m/s
dist = abs(offset*v)
print(f'Time offset (sensor 2 lags sensor 1): {round(-offset, 2)}s')
print(f'Distance between sensors: {round(dist, 2)}m')
