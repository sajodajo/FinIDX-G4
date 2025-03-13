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
    f"<h1 style='font-size: 55px; text-align: center; color: blue;'>{companyName.title()} Share Price Analysis</h1>", 
    unsafe_allow_html=True
)

## GET DATA ##
df = psf.getStockPrices(ticker,start_date,end_date)
df = ARCHLib.calcColumns(df)


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Time Series", "Log Returns", "Normal Distribution","T Distribution","GARCH","Volatility","Residual Analysis","VaR Analysis","Expected Shortfall","Dynamic Risk Modelling"])

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
   ## FIT NORMAL DIST ##
    st.header("Normal Distribution")
    fig = ARCHLib.fitNormalDist(df,companyName)
    st.pyplot(fig)

with tab4:
    ## FIT T DIST ##
    st.header("T Distribution")
    fig = ARCHLib.fitTDist(df,companyName)
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

st.header("About the Project")