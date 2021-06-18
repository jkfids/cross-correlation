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
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numba import njit

# Local modules
from crosscorrelation import norm_corr

# Colour dictionary for category colours
colordict = {
    'Health Care': 'tab:red',
    'Industrials': 'tab:gray',
    'Consumer Discretionary': 'tab:green',
    'Information Technology': 'tab:blue',
    'Consumer Staples': 'tab:orange',
    'Utilities': 'tab:pink',
    'Financials': 'darkgray',
    'Materials': 'tab:cyan',
    'Real Estate': 'tab:olive',
    'Energy': 'tab:brown',
    'Communication Services': 'tab:purple'
}


@njit
def norm_corr2(f, g):
    """
    Normalised correlation function that adjusts length of input vectors if 
    they are not equal length
    """
    try:
        return norm_corr(f, g)
    except:
        size = min(f.size, g.size)
        return norm_corr(f[-size:], g[-size:])


class StockCorr:
    """Stock correlation network class"""

    def __init__(self, companies):
        self.companies = pd.DataFrame(
            companies, columns=['Symbol', 'Name', 'Category'])
        self.companies = self.companies.sort_values(
            by='Symbol', ignore_index=True)

        self.categories = self.companies['Category'].unique().tolist()
        self.symbols = self.companies['Symbol'].tolist()

        self.pricehistdict = {}
        self.max_pricehist_len = 0
        self.categ_pricehistdict = {}
        self.corrmatrix = np.array([])
        self.categ_corrmatrix = np.array([])
        self.corrmatrixdf = pd.DataFrame()
        self.categ_corrmatrixdf = pd.DataFrame()

        self.stocknetwork = nx.Graph()

    def gen_pricehistdict(self, pricetype='Close'):
        """
        Generate a dictionary of price histories from csv data where symbols 
        are keys and values are price histories
        """
        for symb in self.symbols:
            df = pd.read_csv(f"data/stocks/{symb}.csv").dropna()
            pricehist = df[pricetype].to_numpy()
            self.pricehistdict[symb] = pricehist
            # Maximum price history length
            if pricehist.size > self.max_pricehist_len:
                self.max_pricehist_len = pricehist.size
        return self.pricehistdict

    def calc_corrmatrix(self):
        """
        Calculate the correlation matrix between all stocks
        """
        N = len(self.symbols)
        self.corrmatrix = np.zeros((N, N), dtype=np.float64)
        np.fill_diagonal(self.corrmatrix, 1)
        # Iterate over each unique pair of stocks
        for i, symb1 in enumerate(self.symbols):
            for j, symb2 in enumerate(self.symbols):
                if i < j:
                    hist1 = self.pricehistdict[symb1]
                    hist2 = self.pricehistdict[symb2]
                    self.corrmatrix[i, j] = norm_corr2(hist1, hist2)
        self.corrmatrix += self.corrmatrix.T - np.identity(N)

        index = pd.MultiIndex.from_frame(
            sp500.companies[['Category', 'Symbol']])
        self.corrmatrixdf = pd.DataFrame(
            sp500.corrmatrix, index=index, columns=index)
        return self.corrmatrix, self.corrmatrixdf

    def calc_categ_corrmatrix(self):
        """
        Calculate the correlation matrix between category averages
        """
        N = len(self.categories)
        self.categ_corrmatrix = np.zeros((N, N), dtype=np.float64)
        np.fill_diagonal(self.categ_corrmatrix, 1)

        # Calculate the compound average returns for each category
        for categ in self.categories:
            companies = self.companies[self.companies['Category'] == categ]
            total = np.zeros(self.max_pricehist_len)
            for symb in companies['Symbol'].to_list():
                hist = self.pricehistdict[symb]
                ones = np.ones(self.max_pricehist_len)
                ones[-hist.size:] = hist/hist[0]
                total += ones
            self.categ_pricehistdict[categ] = total/len(companies)

        # Iterate over each unique pair of categories to calculate corr matrix
        for i, categ1 in enumerate(self.categories):
            for j, categ2 in enumerate(self.categories):
                if i < j:
                    hist1 = self.categ_pricehistdict[categ1]
                    hist2 = self.categ_pricehistdict[categ2]
                    self.categ_corrmatrix[i, j] = norm_corr2(hist1, hist2)
        self.categ_corrmatrix += self.categ_corrmatrix.T - np.identity(N)

        self.categ_corrmatrixdf = pd.DataFrame(sp500.corrmatrix,
                                               index=self.companies,
                                               columns=self.companies)
        return self.categ_corrmatrix, self.categ_corrmatrixdf

    def gen_stocknetwork(self):
        """
        Generate the stock correlation network as a minimum spanning tree
        """
        G = nx.Graph()
        for i, symb1 in enumerate(self.symbols):
            for j, symb2 in enumerate(self.symbols):
                if i < j:
                    weight = np.sqrt(2*(1 - self.corrmatrix[i][j]))
                    G.add_edge(symb1, symb2, weight=weight)

        T = nx.minimum_spanning_tree(G)
        self.stocknetwork = T
        return self.stocknetwork

    def draw_corrmatrix(self, mode='stock'):
        """Draw either the stock or category correlation matrix"""
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))

        if mode == 'stock':
            im = ax.imshow(self.corrmatrix, vmin=-1, vmax=1)
            fig.colorbar(im, fraction=0.046, pad=0.04)
            ax.set_xlabel('Companies')
            ax.set_ylabel('Companies')
            return fig
        elif mode == 'category':
            im = ax.imshow(self.categ_corrmatrix, vmin=0, vmax=1)
            ax.set_xticks(range(len(self.categories)))
            ax.set_yticks(range(len(self.categories)))
            ax.set_xticklabels(self.categories, rotation=45,
                               horizontalalignment='right', fontsize=8)
            ax.set_yticklabels(self.categories, fontsize=8)
            # Annotate grid
            for i in range(len(self.categories)):
                for j in range(len(self.categories)):
                    ax.text(j, i, round(self.categ_corrmatrix[i, j], 2),
                            ha="center", va="center", color="k", fontsize=8)
            fig.colorbar(im, fraction=0.046, pad=0.04)
            return fig

    def draw_network(self, colordict=colordict):
        """Draw the stock correlation network in the kamada kawai layout"""
        node_color = []
        for category in self.companies['Category'].to_list():
            node_color.append(colordict[category])
        edge_color = [self.stocknetwork[u][v]['weight']
                      for u, v in self.stocknetwork.edges]

        fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
        pos = nx.kamada_kawai_layout(self.stocknetwork)
        # Draw nodes with colordict
        nodes = nx.draw_networkx_nodes(self.stocknetwork, pos,
                                       node_color=node_color,
                                       node_size=10, ax=ax)
        # Draw edges with colormap
        edges = nx.draw_networkx_edges(self.stocknetwork, pos,
                                       edge_color=edge_color,
                                       edge_cmap=plt.cm.viridis,
                                       alpha=0.7, ax=ax)
        # Create legend
        handles = []
        for key, value in colordict.items():
            handles.append(Line2D([0], [0], marker='o', color='w', label=key,
                                  markerfacecolor=value, markersize=7.5))
        fig.colorbar(edges, fraction=0.025, pad=0.01)
        ax.set_axis_off()
        ax.legend(handles=handles, ncol=1, fontsize=7.5, loc='lower left'  )
        return fig


if __name__ == "__main__":
    # Load generated stock data from csv file and create class
    print('Loading stock data...')
    infodf = pd.read_csv("data/S&P500_info.csv")
    companies = infodf[['Symbol', 'Security', 'GICS Sector']].values.tolist()
    sp500 = StockCorr(companies)
    sp500.gen_pricehistdict()

    # Calculate and plot the cross-correlation matrix
    print('Calculating stock cross-correlation matrix...')
    start = time()
    sp500.calc_corrmatrix()
    end = time()
    print(f'Time elapsed: {round(end - start, 2)}s')
    fig1 = sp500.draw_corrmatrix('stock')
    fig1.tight_layout()
    fig1.savefig("output/stock_corrmatrix")

    # Calculate and plot the cross-correlation matrix
    print('Calculating category cross-correlation matrix...')
    start = time()
    sp500.calc_categ_corrmatrix()
    end = time()
    print(f'Time elapsed: {round(end - start, 2)}s')
    fig2 = sp500.draw_corrmatrix('category')
    fig2.tight_layout()
    fig2.savefig("output/categ_corrmatrix")

    # Generate the stock correlation network as a minimum spanning tree
    print('Generating stock correlation network...')
    start = time()
    T = sp500.gen_stocknetwork()
    end = time()
    print(f'Time elapsed: {round(end - start, 2)}s')

    # Draw and save the network
    print('Drawing graph...')
    node_color = []

    fig3 = sp500.draw_network()
    fig3.tight_layout()
    fig3.savefig("output/stocknetwork.png")
    total_weight = T.size(weight='weight')
    print(f'Normalised Tree Length: {total_weight/T.size()}')
