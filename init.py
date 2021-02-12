#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import pandas as pd

# stocks = ["AAPL", "NVDA", "KO", "UL", "BAC", "AXP"]
# azioni rispettivamente: Apple, Nvidia, CocaCola, Unilever, Bank of america, America Express
# tutte le azioni sono americane (NasdaqGS (mercato elettronico) e New York Stock Exchange)

# serve per le stampe
counter = 1

def download_data(stocks = ["AAPL", "NVDA", "KO", "UL", "BAC", "AXP"], start_stream = '2015-01-01', end_stream = '2020-10-01'):
    # dictionary witch contains the name and the dataframe
    dataframes = {}
    
    # adds stock's dataframe in dataframes dictionary
    for s in stocks:
        print("[] downloading " + s + "\t -> ", end = "")
        dataframes[s] = web.get_data_yahoo(s, start_stream, end_stream)
        print("Done")
    
    return dataframes

def plot_line(df = {}, title = "", xlabel = "", ylabel = "", grid = True, xlines=[], save = False):
    plt.figure(figsize=(14, 9)) # set dpi=1200 for HQ images
    plt.title(title)
    for d in df:
        plt.plot(df[d], label=d)
    if grid == True:
        plt.grid()
    colours = ["g", "m", "y", "c"]
    for line in xlines:
        count = count + 1
        plt.axvline(x = line, linewidth = 2, color = colours[count-(len(colours)*(count//len(colours)))])
        # count-(len(colors)*(count//len(colours))) prende ogni volta un nuovo colore dalla lista
    plt.legend()
    if save == True:
        global counter
        plt.savefig('img/'+ str(counter) + '-' +title+'.png', dpi=300)
        counter += 1
    return plt

def plot_hist(df, zeroline = False, dens = False, norma = False, title = "", xlabel = "returns", ylabel = "", bins = 10, grid = True, xlines=[]):
    plt.figure(figsize=(14, 9))
    plt.hist(df, density = True, bins=bins)
    if dens == True:
        df.plot.density(label = "CC Returns distribution")
    if norma == True:
        # doesn't work, todo
        from scipy.stats import norm
        mu, std = norm.fit(df)
        plt.hist(df, density=True, alpha=0.6, color='g')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    count = 0   # serve per i colori
    colours = ["g", "m", "y", "c"]
    for line in xlines:
        count = count + 1
        plt.axvline(x = line, linewidth = 2, color = colours[count-(len(colours)*(count//len(colours)))])
        # count-(len(colors)*(count//len(colours))) prende ogni volta un nuovo colore dalla lista
    if zeroline == True:
        # draws a vertical line on the returns = 0 to separate negative values from positive ones
        plt.axvline(x=0, linewidth=2, color='y', label = "return = 0")
    plt.legend()
    return plt


# box plot
def plot_box(df, title = "", grid = True):
    plt.figure(figsize=(10, 9))
    plt.boxplot(df)
    plt.title(title)
    if grid == True:
        plt.grid()
    return plt



