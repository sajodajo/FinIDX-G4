import pandas as pd
import seaborn as sns
import simfin as sf
from simfin.names import *
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
import os
import pandas as pd


class pySimFin:

    def __init__(self):
            load_dotenv()  
            self.api_key = os.getenv('API_KEY') 
            if not self.api_key:
                raise ValueError("API_KEY not found in environment variables")

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
    
    def get_companies(self,period='annual'):
        df_statements = sf.load_income(variant=period).sort_index()
        df_statements['Ticker'] = df_statements.index.get_level_values(0)
        return list(df_statements['Ticker'].unique())
    
    def getCompanyInfo(self,ticker):
        load_dotenv()
        API_KEY = os.getenv('API_KEY')
        headers = {
            "accept": "application/json",
            "Authorization": API_KEY
        }
        url = "https://backend.simfin.com/api/v3/companies/general/compact?ticker=" + ticker
        response = requests.get(url, headers=headers).json()

        return pd.DataFrame(response['data'], columns=response['columns'])
    

    

