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
companyName = st.sidebar.selectbox("Select a company:", companydf['name'])
ticker = companydf.loc[companydf['name'] == companyName, 'ticker'].values[0]

minDate = '2019-04-15'
today1yrAgo = datetime.date.today() - datetime.timedelta(days=1)
try:
    start_date, end_date = st.sidebar.date_input("Select a date range", [minDate, today1yrAgo],min_value=minDate,max_value=today1yrAgo)
except (ValueError,TypeError,NameError):
    pass 


## HEADER ##
st.header(f'{companyName.title()} Share Price Analysis')

## GET DATA ##
df = psf.getStockPrices(ticker,start_date,end_date)
df = ARCHLib.calcColumns(df)

## TIME SERIES ##
st.header("Time Series")
fig, ax = ARCHLib.plotTS(df,companyName)
st.pyplot(fig)

## VISUALIZE LOG RETURNS ##
st.header("Log-Returns")
fig, ax = ARCHLib.plotLR(df, ARCHLib.smoothed_abs,companyName)
st.pyplot(fig)

## FIT NORMAL DIST ##
st.header("Normal Distribution")
fig = ARCHLib.fitNormalDist(df,companyName)
st.pyplot(fig)

## FIT NORMAL DIST ##
st.header("T Distribution")
fig = ARCHLib.fitTDist(df,companyName)
st.pyplot(fig)

## FIT GARCHLib ##
st.header("GARCH")
fig, returns, garch_fit = ARCHLib.fitGARCH(df)
st.pyplot(fig)

## VISUALIZE VOLATILITY ##
st.subheader("GARCH - Volatility")
fig, ax = ARCHLib.visVolatility(df, returns, garch_fit)
st.pyplot(fig)

## RESIDUAL ANALYSIS ##
st.subheader("GARCH - Residual Analysis")
fig, ax = ARCHLib.residualAnalysis(garch_fit)
st.pyplot(fig)

## VaR ANALYSIS ##
st.subheader("Risk Modelling - VaR Analysis")
fig, confidence_levels = ARCHLib.VaR(df)
st.pyplot(fig)

## EXPECTED SHORTFALL ##
st.subheader("Risk Modelling - Expected Shortfall")
fig = ARCHLib.expectedShortfall(confidence_levels)
st.pyplot(fig)

## DYNAMIC RISK MODELLING ##
st.subheader("Dynamic Risk Modelling")
fig = ARCHLib.dynamicRM(garch_fit)
st.pyplot(fig)