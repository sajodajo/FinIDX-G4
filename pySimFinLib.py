import pandas as pd
import seaborn as sns
import simfin as sf
from simfin.names import *


class pySimFin:

    def __init__(self):
        self.api_key = '70d5d920-9f9e-4062-9311-1b4df7c98ba4'
        sf.set_data_dir('~/simfin_data/')
        sf.load_api_key(path='~/simfin_api_key.txt')
        sf.set_api_key(api_key='70d5d920-9f9e-4062-9311-1b4df7c98ba4')
        sns.set_style("whitegrid")

    def get_share_prices(self,tickers,start,end):       
        hub = sf.StockHub(market='us', tickers=tickers,
                        refresh_days_shareprices=1)

        df_prices = hub.load_shareprices(variant='daily').sort_index()
        df_prices = df_prices.loc[start:end]
        return df_prices
    
    def get_financial_statement(self,tickers, start, end,period='quarterly'):
        df_statements = sf.load_income(variant=period).sort_index()
        idx = pd.IndexSlice
        df_statements = df_statements.loc[idx[tickers, start:end], :]
        return df_statements