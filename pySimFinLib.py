import pandas as pd
import seaborn as sns
import simfin as sf
from simfin.names import *
import yfinance as yf
import matplotlib.pyplot as plt


class pySimFin:

    def __init__(self):
        self.api_key = '70d5d920-9f9e-4062-9311-1b4df7c98ba4'
        sf.set_data_dir('~/simfin_data/')
        sf.load_api_key(path='~/simfin_api_key.txt')
        sf.set_api_key(api_key='70d5d920-9f9e-4062-9311-1b4df7c98ba4')
        sns.set_style("whitegrid")

    def getTickerMap(self):
        import requests
        import pandas as pd

        url = "https://raw.githubusercontent.com/dhhagan/stocks/master/scripts/stock_info.csv"
        response = requests.get(url)

        with open("tickerMap.csv", "wb") as file:
                file.write(response.content)

        tickerMap = pd.read_csv("tickerMap.csv")
        return tickerMap

    def tickerMatch(self,input_string):
        tickerMap = self.getTickerMap()
        for _, company in tickerMap.iterrows():
            if input_string.lower() in company['Name'].lower(): 
                return company['Ticker']  
        return None  

    def get_share_prices(self,companies:list,start,end):

        tickers = list(map(lambda x: self.tickerMatch(x), companies))   
          
        hub = sf.StockHub(market='us', tickers=tickers,
                        refresh_days_shareprices=1)

        df_prices = hub.load_shareprices(variant='daily').sort_index()
        df_prices = df_prices.loc[pd.IndexSlice[:, start:end], :]
        return df_prices
    
    def get_financial_statement(self,companies, start, end,period='quarterly'):
        tickers = list(map(lambda x: self.tickerMatch(x), companies))  
        df_statements = sf.load_income(variant=period).sort_index()
        idx = pd.IndexSlice
        df_statements = df_statements.loc[idx[tickers, start:end], :]
        return df_statements
    

    

