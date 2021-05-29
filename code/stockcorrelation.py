# -*- coding: utf-8 -*-
"""
Created on Sat May 29 11:03:09 2021

@author: Fidel
"""

# Standard libraries
from time import time
import pandas as pd
import numpy as np

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from crosscorrelation import *

infodf = pd.read_csv("data/S&P500_info.csv")

# Generate dictionary of stock price histories
companies = infodf['Symbol'].values.tolist()
stock_dict = {}
for company in companies[:3]:
    stockdf = pd.read_csv(f"data/stocks/{company}.csv")
    price_hist = stockdf['Close'].to_numpy()
    stock_dict[company] = price_hist
    
stock_dict['MMM']

r = norm_crosscorr(stock_dict['MMM'], stock_dict['ABBV'])

class StockCorr:
    def __init__(self):
        pass
    