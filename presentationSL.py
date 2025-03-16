import streamlit as st
from pySimFinLib import pySimFin
import datetime
import ARCHLib
import numpy as np
import arch
from psf2 import pySimFin2


st.set_page_config(layout = 'wide')


psf = pySimFin2()

## SIDEBAR SELECTORS ##
st.sidebar.title("Filter")

companydf = psf.getCompanyList()
companyName = st.sidebar.selectbox("Select a company:", companydf['name'].sort_values())
ticker = companydf.loc[companydf['name'] == companyName, 'ticker'].values[0]

minDate = '2019-04-15'
today1yrAgo = datetime.date.today() - datetime.timedelta(days=1)
try:
    start_date, end_date = st.sidebar.date_input("Select a date range", [minDate, today1yrAgo],min_value=minDate,max_value=today1yrAgo)
except (ValueError,TypeError,NameError):
    pass 


## Title ##
st.markdown(
    f"<h1 style='font-size: 60px; text-align: center; color: #389cfc;'>{companyName.title()} Share Price Analysis</h1>", 
    unsafe_allow_html=True
)

st.markdown(
    f"<h2 style='font-size: 45px; text-align: center; color:black'> 1. About the Dataset </h2>", 
    unsafe_allow_html=True
)

st.markdown('''
<p style='font-size:18px; text-align:justify; color:black'>
This project is a dashboard using to analyze the stock price of a company. The dashboard uses ARCH and GARCH models to analyze the stock price data. ARCH and GARCH models are used to model the volatility of the stock price data. The dashboard also includes a VaR analysis and Expected Shortfall analysis to model the risk of the stock price data. The dashboard is built using Streamlit and Python.
</p>

<p style='font-size:18px; text-align:justify; color:black'>
The selected dataset is sourced from SimFin (Simplifying Finance), a financial data provider that offers standardized financial statements, market data, and derived metrics for publicly traded companies. The dataset includes income statements, balance sheets, cash flow statements, stock prices, and fundamental financial ratios for a wide range of companies, primarily from U.S. stock markets.
</p>

<p style='font-size:18px; text-align:justify; color:black'><b>Source:</b>
The data is obtained from SimFin’s open financial database, which compiles financial reports from SEC filings (10-K, 10-Q reports) and other public sources. SimFin standardizes the raw financial data, ensuring consistency and ease of use for financial analysis.
</p>
''', unsafe_allow_html=True)

info = psf.getCompanyInfo(ticker)['companyDescription'].values[0]
industry = psf.getCompanyInfo(ticker)['industryName'].values[0]

# Second markdown with dynamic values
st.markdown(f'''
<p style='font-size:20px; text-align:center; color:green'>
<b>The company selected is {companydf.loc[companydf["name"] == companyName, "name"].values[0]} with the ticker '{ticker}'. {info} The company is in the {industry} industry. The data is from {start_date} to {end_date}.</b>
</p>
''', unsafe_allow_html=True)

## GET DATA ##
df = psf.getStockPrices(ticker,start_date,end_date)
df = ARCHLib.calcColumns(df)


st.markdown(
    f"<br><h2 style='font-size: 45px; text-align: center; color:black'> 2. Exploratory Data Analysis </h2>", 
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Log Returns", "Summary Statistics & Stationarity","Autocorrelation"])

with tab1:
    ## TIME SERIES ##
    fig, ax = ARCHLib.plotTS(df,companyName)
    st.pyplot(fig)

with tab2:
    ## VISUALIZE LOG RETURNS ##
    st.header("Log-Returns")
    fig, ax = ARCHLib.plotLR(df, ARCHLib.smoothed_abs,companyName)
    st.pyplot(fig)

with tab3:
    'Hello'
    st.dataframe(ARCHLib.adf_test(df))

with tab4:
    'Hello'

st.markdown(
    f"<br><br><h2 style='font-size: 45px; text-align: center; color:black'> 3. Volatility Modelling </h2>", 
    unsafe_allow_html=True
)

st.markdown(
    f"<br><br><h2 style='font-size: 45px; text-align: center; color:black'> 4. Risk Modelling </h2>", 
    unsafe_allow_html=True
)

st.markdown(
    f"<br><br><h2 style='font-size: 45px; text-align: center; color:black'> 5. Conclusions </h2>", 
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Time Series", "Log Returns", "Normal Distribution","T Distribution","GARCH","Volatility","Residual Analysis","VaR Analysis","Expected Shortfall","Dynamic Risk Modelling"])





with tab3:
   ## FIT NORMAL DIST ##
    st.header("Normal Distribution")
    fig, statsDict = ARCHLib.fitNormalDist(df,companyName)
    st.pyplot(fig)
    st.dataframe(statsDict)

with tab4:
    ## FIT T DIST ##
    st.header("T Distribution")
    ARCHLib.fitTDist(df,companyName)

    fig = ARCHLib.plotFitTDist(df,companyName)
    st.pyplot(fig)

with tab5:
    ## FIT GARCHLib ##
    st.header("GARCH")
    fig, returns, garch_fit = ARCHLib.fitGARCH(df)
    st.pyplot(fig)

with tab6:
    ## VISUALIZE VOLATILITY ##
    st.subheader("GARCH - Volatility")
    fig, ax = ARCHLib.visVolatility(df, returns, garch_fit)
    st.pyplot(fig)

with tab7:
    ## RESIDUAL ANALYSIS ##
    st.subheader("GARCH - Residual Analysis")
    fig, ax = ARCHLib.residualAnalysis(garch_fit)
    st.pyplot(fig)

with tab8:
    ## VaR ANALYSIS ##
    st.subheader("Risk Modelling - VaR Analysis")
    fig, confidence_levels = ARCHLib.VaR(df,companyName)
    st.pyplot(fig)

with tab9:
    ## EXPECTED SHORTFALL ##
    st.subheader("Risk Modelling - Expected Shortfall")
    fig = ARCHLib.expectedShortfall(confidence_levels,companyName)
    st.pyplot(fig)

with tab10:
    ## DYNAMIC RISK MODELLING ##
    st.subheader("Dynamic Risk Modelling")
    fig = ARCHLib.dynamicRM(garch_fit,companyName)
    st.pyplot(fig)

