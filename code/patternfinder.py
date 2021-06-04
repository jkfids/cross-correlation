# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 23:45:58 2021

@author: Fidel
"""

import numpy as np
from scipy.io.wavfile import read, write

from crosscorrelation import spectral_crosscorr, calc_offset

song = read("data/Unfolding.wav")
sampling_rate = song[0]
song = song[1]


snare = song[round(30.25*sampling_rate)+1600:round(30.6*sampling_rate)]
#snare = snare[1600:5000]
write("data/snare.wav", sampling_rate, snare)
snare_padded = np.pad(snare, (0, song.size-snare.size))

r = spectral_crosscorr(song, snare_padded)
offset = calc_offset(r, 1/sampling_rate, mode='spectral')

write("data/test.wav", sampling_rate, song[3924142-snare.size*5:3924142+snare.size*5])
