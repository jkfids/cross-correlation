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
import matplotlib.pyplot as plt
from numba import njit
import seaborn as sns

# Local modules
from crosscorrelation import norm_corr

@njit
def norm_corr2(f, g):
    """"""
    try:
        return norm_corr(f, g)
    except:
        size = min(f.size, g.size)
        return norm_corr(f[-size:], g[-size:])


class StockCorr:
    def __init__(self, companies):
        self.companies = pd.DataFrame(companies, columns=['Symbol', 'Name', 'Category'])
        self.companies = self.companies.sort_values(by=['Category', 'Symbol'], ignore_index=True)
        
        self.categories = self.companies['Category'].unique().tolist()
        self.symbols = self.companies['Symbol'].tolist()
        
        self.pricehistdict = {}
        self.corrmatrix = np.array([])
        self.corrmatrixdf = pd.DataFrame()
            
    def gen_pricehistdict(self, pricetype='Close'):
        """"""
        for symb in self.symbols:
            df = pd.read_csv(f"data/stocks/{symb}.csv").dropna()
            pricehist = df[pricetype].to_numpy()
            self.pricehistdict[symb] = pricehist
        return self.pricehistdict
            
    def gen_corrmatrix(self):
        """"""
        N = len(self.symbols)
        self.corrmatrix = np.zeros((N, N), dtype=np.float64)
        np.fill_diagonal(self.corrmatrix, 1)
        for i, symb1 in enumerate(self.symbols):
            for j, symb2 in enumerate(self.symbols):
                if i < j:
                    pricehist1 = self.pricehistdict[symb1]
                    pricehist2 = self.pricehistdict[symb2]
                    self.corrmatrix[i, j] = norm_corr2(pricehist1, pricehist2)
        self.corrmatrix += self.corrmatrix.T - np.identity(N)
        
        index = pd.MultiIndex.from_frame(sp500.companies[['Category', 'Symbol']])
        self.corrmatrixdf = pd.DataFrame(sp500.corrmatrix, index=index, columns=index)
        return self.corrmatrix, self.corrmatrixdf   
        
if __name__ == "__main__":
    
    infodf = pd.read_csv("data/S&P500_info.csv")
    companies = infodf[['Symbol', 'Security', 'GICS Sector']].values.tolist()
    
    sp500 = StockCorr(companies)
    sp500.gen_pricehistdict()
    sp500.gen_corrmatrix()
    plt.imshow(sp500.corrmatrix, vmin=-1, vmax=1)
    plt.colorbar()
    ticks = [0 for i in range(len(sp500.categories))]
    sumi = 0
    for i in range(len(sp500.categories)):
        ticks[i] = ((sp500.companies['Category']==sp500.categories[i]).sum()/2+sumi)
        sumi += (sp500.companies['Category']==sp500.categories[i]).sum()
    plt.yticks(ticks, sp500.categories)
    
    
    
    
    
        