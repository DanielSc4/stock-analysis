#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:50:12 2020

@author: daniel
"""

import numpy as np
from numpy.lib.function_base import cov
import pandas as pd
import matplotlib.pyplot as plt

from init import download_data, plot_line, plot_hist, plot_box


def get_AdjC(df):
    # returns a dataframe with 1 column: adj closes
    df_adj = df["Adj Close"]
    return df_adj

def compute_CCreturns(df, aggregation = "M", name=""):
    df_adj = df["Adj Close"]    # using only adj closes
    df_adj_op = df_adj.groupby(pd.Grouper(freq = aggregation))  # aggregation (default: monthly)
    df_adj = df_adj_op.mean()   # aggregate with avg
    df_returns = np.log(df_adj/df_adj.shift(1)) # calculating CC returns
    df_returns.name = name + " CC Return"
    
    # dropping na values
    return df_returns.dropna()


def compared_CCreturns(stocks_in = []):
    # s is the stock's id
    # dfs contains all the dataframe that need to be compered
    dfs = {}
    for s in stocks_in:
        dfs[s] = compute_CCreturns(dataframes[s])
    
    return plot_line(dfs, title = "Continuous Compound of " + str(stocks_in), xlabel = "time", ylabel = "value")


def descriptive_stats(df):
    ret = {}
    ret["mean"] = df.mean()
    ret["var"] = df.var()
    ret["std"] = df.std()
    ret["min"] = df.min()
    ret["max"] = df.max()
    ret["skew"] = df.skew()             # simmetry
    # skewness -> measures symmetry of a distribution around its mean
    #             = 0 symmetric (normal distribution)
    #             > 0 longer right tail than normal distribution
    #             < 0 longer left tail than normal distribution
    ret["kurt"] = df.kurtosis()     # thickness
    # kurtosis -> measures tail thickness of distribution
    #             > 0 tail fatter than normal distribution tail
    #             < 0 tail thinner than normal distribution tail
    #             = 0 tail as normal distribution tail
    ret["0.25"] = df.describe()[4]
    ret["0.50"] = df.describe()[5]  # median value
    ret["0.75"] = df.describe()[6]

    return ret


def main():
    
    ################################
    ################################
    #### only for debug purpose ####
    ################################
    ################################
    
    
    # stocks = ["AAPL", "NVDA", "KO", "UL", "BAC", "AXP"]
    dataframes = download_data()

    # plotting adjusted closes individually
    print("-- plotting adjusted closes individually")
    for s in ["AAPL", "NVDA", "KO", "UL", "BAC", "AXP"]:
        plot_line({s: get_AdjC(dataframes[s])}, title = s + " adj closes", xlabel = "time", ylabel = "value").show()

    
    # comparing all the stock's adj close in the list simultaneously
    print("-- comparing all the stock's adj close in the list simultaneously")
    dfs = {}
    for s in ["AAPL", "NVDA", "KO", "UL", "BAC", "AXP"]:
        dfs[s] = get_AdjC(dataframes[s])
    plot_line(dfs, title = "Adj Closes", xlabel = "time", ylabel = "value").show()


    # comparing stocks according to their industry:
    print("-- comparing stocks according to their industry")
    compared_CCreturns(["AAPL", "NVDA"])    # tech
    compared_CCreturns(["KO", "UL"])        # food
    compared_CCreturns(["BAC", "AXP"])      # bank


    # Distribution CC returns
    print("-- showing distribution CC returns")
    for s in ["AAPL", "NVDA", "KO", "UL", "BAC", "AXP"]:
        plot_hist(compute_CCreturns(dataframes[s]), zeroline = True, dens = True, title = "Distribution of CC return about " + s + " stock").show()
    # histogram for diagnostic analysis
    # il rischio è più basso quanto più i ritorni assumo la forma di una normale
    # idea: generare una normale per comparare il risultato

    # to-do: possibility to generate a subplot
    
    # univariate descriptive stats
    print("-- printing univariate descriptive statistics data")
    for s in ["AAPL", "NVDA", "KO", "UL", "BAC", "AXP"]:
        stats = descriptive_stats(compute_CCreturns(dataframes[s]))
        print("Descriptive statistics about " + s)
        for ele in stats:
            print("\t" + ele + "\t" + str(stats[ele]))

    print("-- plotting min, max and quantiles in returns distribution")
    for s in ["AAPL", "NVDA", "KO", "UL", "BAC", "AXP"]:
        return_stock = compute_CCreturns(dataframes[s])
        stat = descriptive_stats(return_stock)
        plot_hist(return_stock, dens = False, title = s, bins=20, xlines = list([stat["min"], stat["0.25"], stat["0.50"], stat["0.75"], stat["max"]])).show()

        
    print("-- comparing gaussian distribution quantiles (red line) with stocks ones (qqplot) (blue points)")
    # la lontananza aumenta il valore di kurtosis
    import statsmodels.api as sm
    for s in ["AAPL", "NVDA", "KO", "UL", "BAC", "AXP"]:
        sm.qqplot(compute_CCreturns(dataframes[s]), line = 's')
        plt.show()
    

    print("-- showing box plots of all shares")
    for s in ["AAPL", "NVDA", "KO", "UL", "BAC", "AXP"]:
        plot_box(compute_CCreturns(dataframes[s]), title = s + " box plot").show()


    print("-- comparing boxplots of the same financial sectors")
    data = pd.concat([compute_CCreturns(dataframes["AAPL"]), compute_CCreturns(dataframes["NVDA"])], axis=1)
    data.columns = ["AAPL", "NVDA"]
    data.boxplot()
    plt.show()
    data = pd.concat([compute_CCreturns(dataframes["KO"]), compute_CCreturns(dataframes["UL"])], axis=1)
    data.columns = ["KO", "UL"]
    data.boxplot()
    plt.show()
    data = pd.concat([compute_CCreturns(dataframes["BAC"]), compute_CCreturns(dataframes["AXP"])], axis=1)
    data.columns = ["BAC", "AXP"]
    data.boxplot()
    plt.show()
    
# end main



# Start here



# TO-DO: da cambiare il != con == dopo aver dinito la fase di descriptive analytics
# no lascia perdere
if __name__ == "__main__":
    main()





