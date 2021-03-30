# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:50:44 2021

@author: Fidel
"""

import numpy as np

def appendt():
    a = np.array([])
    for i in range(10):
        a = np.append(a, i)
    return a

def assignt():
    a = np.zeros(10)
    for i in range(10):
        a[i] = i
    return a