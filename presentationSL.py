import streamlit as st

st.set_page_config(layout = 'wide')
                   

from pySimFinLib import pySimFin
import datetime
import ARCHLib as arch
import numpy as np

psf = pySimFin()

## SIDEBAR SELECTORS ##
st.sidebar.title("Filter")

companydf = psf.getTickerMap()[psf.getTickerMap()['Ticker'].isin(psf.get_companies())]
companyName = st.sidebar.selectbox("Select a company:", companydf['Name'])
ticker = companydf.loc[companydf['Name'] == companyName, 'Ticker'].values[0]


companies = list([companyName])
minDate = '2019-04-15'
today1yrAgo = datetime.date.today() - datetime.timedelta(days=365)
try:
    start_date, end_date = st.sidebar.date_input("Select a date range", [minDate, today1yrAgo],min_value=minDate,max_value=today1yrAgo)
except (ValueError,TypeError,NameError):
    pass 

## HEADER ##
st.header(f'{companyName.title()} Share Price Analysis')

## GET DATA ##
df = psf.get_share_prices(companies,start_date,end_date)
df = df.droplevel(0)
df = df.drop(columns='Dividend')
df = arch.calcColumns(df)

## TIME SERIES ##
st.header("Time Series")
fig, ax = arch.plotTS(df,companies)
st.pyplot(fig)

## VISUALIZE LOG RETURNS ##
st.header("Log-Returns")
fig, ax = arch.plotLR(df, arch.smoothed_abs,companies)
st.pyplot(fig)

## FIT NORMAL DIST ##
st.header("Normal Distribution")
fig = arch.fitNormalDist(df,companies)
st.pyplot(fig)

## FIT NORMAL DIST ##
st.header("T Distribution")
fig = arch.fitTDist(df,companies)
st.pyplot(fig)

## FIT GARCH ##
st.header("GARCH")
fig, returns, garch_fit = arch.fitGARCH(df)
st.pyplot(fig)

## VISUALIZE VOLATILITY ##
st.subheader("GARCH - Volatility")
fig, ax = arch.visVolatility(df, returns, garch_fit)
st.pyplot(fig)

## RESIDUAL ANALYSIS ##
st.subheader("GARCH - Residual Analysis")
fig, ax = arch.residualAnalysis(garch_fit)
st.pyplot(fig)

## VaR ANALYSIS ##
st.subheader("Risk Modelling - VaR Analysis")
fig, confidence_levels = arch.VaR(df)
st.pyplot(fig)

## EXPECTED SHORTFALL ##
st.subheader("Risk Modelling - Expected Shortfall")
fig = arch.expectedShortfall(confidence_levels)
st.pyplot(fig)

## DYNAMIC RISK MODELLING ##
st.subheader("Dynamic Risk Modelling")
fig = arch.dynamicRM(garch_fit)
st.pyplot(fig)