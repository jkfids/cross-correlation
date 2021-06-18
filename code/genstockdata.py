# -*- coding: utf-8 -*-
"""
Created on Sat May 29 01:02:17 2021

@author: Fidel
"""

# Standard libraries
from time import time
import pandas as pd
from yfinance import Ticker


# Set price history time period
period = '2y'  
start = ''
end = ''

# Generate a dataframe containing info about the S&P500 and save it as csv
table = pd.read_html(
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
infodf = table[0]
infodf['Symbol']=infodf['Symbol'].str.replace('.', '-', regex=True)
infodf.to_csv("data/S&P500_info.csv", index=False)


# Obtain the price history of each stock and save it as csv
print(f'Obtaining historical prices ({period})...')
companies = infodf[['Symbol', 'Security', 'GICS Sector']].values.tolist()

start = time()
for company in companies:
    symbol, name, categ = company
    symbol.replace('.', '-')
    print(f'{symbol}, {name} ({categ})')
    pricehistdf = Ticker(symbol).history(period=period)
    pricehistdf = pricehistdf.dropna()
    pricehistdf.to_csv(f"data/stocks/{symbol}.csv",
              columns=['Open', 'High', 'Low', 'Close'])
end = time()
print(f'Time elapsed (generate stock data): {round(end - start, 2)}s')
