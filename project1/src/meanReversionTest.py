import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

# https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing-Part-II/

def plotPriceSeries(df: pd.DataFrame) -> None:
    '''
    Plots the price series data of the tickers held in df.
    
    :param df: pandas DataFrame with Adj Close data, pulled from yfinance
    :type df: pd.DataFrame
    '''
    fig = df.plot(title="Price Series")
    fig.set_ylabel("Price($)")
    plt.show()

def plotScatterSeries(df: pd.DataFrame) -> None:
    df.plot.scatter(x=0, y=1, title="Price Scatterplot")
    plt.show()

def fitLinearModel(df: pd.DataFrame) -> tuple:
    '''
    Tries to fit a line to the price scatterplot, calculates beta and the residuals
    
    :param df: Description
    :type df: pd.DataFrame
    '''
    headers = list(df.columns)
    x = df[headers[0]]
    y = df[headers[1]]
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    res = model.fit()

    intercept, beta = res.params
    df['Residuals'] = res.resid

    return df, beta, intercept

def plotResiduals(df: pd.DataFrame) -> None:
    df.plot(y="Residuals", title="Residual Plot")
    plt.ylabel("Price($)")
    plt.show()

def doCadf(df: pd.DataFrame) -> tuple:
    cadf = ts.adfuller(df["Residuals"])
    testStatistic, pValue, criticalValues = cadf[0], cadf[1], cadf[4]
    print(f'Test statistic: {testStatistic}')
    print(f'p value: {pValue}')
    return testStatistic, pValue, criticalValues