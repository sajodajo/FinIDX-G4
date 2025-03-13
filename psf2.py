import pandas as pd
import seaborn as sns
import simfin as sf
from simfin.names import *
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
import os
import pandas as pd



class pySimFin2:

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
    
    def getStockPrices(self,ticker, start_date, end_date):
        load_dotenv()
        API_KEY = os.getenv('API_KEY')
        
        url = f"https://backend.simfin.com/api/v3/companies/prices/compact?ticker={ticker}&start={start_date}&end={end_date}"

        headers = {
            "accept": "application/json",
            "Authorization": API_KEY
        }

        response = requests.get(url, headers=headers).json()

        df_prices = pd.DataFrame(response[0]['data'], columns=response[0]['columns'])

        return df_prices