# -*- coding: utf-8 -*-
"""
Created on Sat May 29 01:02:17 2021

@author: Fidel
"""

# Standard libraries
from time import time
import pandas as pd
from yfinance import Ticker


period = '1y'  # Set price history time period

# Generate a dataframe containing info about the S&P500 and save it as csv
table = pd.read_html(
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
df = table[0]
df.to_csv("data/S&P500_info.csv", index=False)


# Obtain the price history of each stock and save it as csv
companies = df[['Symbol', 'Security', 'GICS Sector']].values.tolist()

start = time()
for company in companies:
    symbol, name, categ = company
    print(f'{symbol}, {name} ({categ})')
    ticker = Ticker(symbol)
    df = ticker.history(period=period)
    df.to_csv(f"data/stocks/{symbol}.csv",
              columns=['Open', 'High', 'Low', 'Close'])
end = time()
print(f'Time elapsed (generate stock data): {round(end - start, 2)}s')
