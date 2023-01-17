import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from datetime import date, timedelta, datetime
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import pickle
from copulas.multivariate import GaussianMultivariate


class DataFactory:
    def get_time_series(self, ticker_file_path: str, start_date, end_date):
        tickers = pickle.load(open(ticker_file_path, "rb"))
        data = [self._get_ticker_info(ticker, start_date, end_date)
                for ticker in tickers.keys()]
        print(data[0].head())

    def _get_ticker_info(self, ticker: str, start, end):
        ticker_data = yf.Ticker(ticker).history(start=start,
                                                end=end,
                                                interval="1d")

        ticker_data.index = pd.to_datetime(ticker_data.index)
        ticker_data.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
        return ticker_data


start_date = datetime(2020, 1, 1)
end_date = date.today().strftime("%Y-%m-%d")

data_factory = DataFactory()

#data_factory.get_time_series("data/swedish_tickers.p", start_date, end_date)

sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})

nasdaq_data = yf.download('^IXIC', start=datetime(2020, 1, 1).date(),
                          end=date.today().strftime("%Y-%m-%d"))

data = nasdaq_data[['Open', 'Volume', 'Adj Close']]
data.index = pd.to_datetime(data.index.date)
print(data.head())
copula = GaussianMultivariate()
copula.fit(data)
samples = copula.sample(1000)

samples.plot(kind="scatter", x="Open", y="Volume")
