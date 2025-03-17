import streamlit as st
import datetime
import ARCHLib
import numpy as np
import arch
import pandas as pd
from psf2 import pySimFin2


st.set_page_config(layout = 'wide')


psf = pySimFin2()

## SIDEBAR SELECTORS ##
st.sidebar.title("Filter")

companydf = psf.getCompanyList()
companyName = st.sidebar.selectbox("Select a company:", companydf['name'].sort_values(),index=94)
ticker = companydf.loc[companydf['name'] == companyName, 'ticker'].values[0]

minDate = '2019-04-15'
today1yrAgo = datetime.date.today() - datetime.timedelta(days=1)
try:
    start_date, end_date = st.sidebar.date_input("Select a date range", [minDate, today1yrAgo],min_value=minDate,max_value=today1yrAgo)
except (ValueError,TypeError,NameError):
    pass 


## Title ##
st.markdown(
    f"<h1 style='font-size: 60px; text-align: left; color: #389cfc;'>{companyName.title()} Share Price Analysis</h1>", 
    unsafe_allow_html=True
)

st.markdown(
    f"<h2 style='font-size: 45px; text-align: left; color:black'> 1. About the Dataset </h2>", 
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

st.markdown(f'''
<p style='font-size:20px; text-align:left; color:green'>
<b>The company selected is {companydf.loc[companydf["name"] == companyName, "name"].values[0]} with the ticker '{ticker}'. {info} The company is in the {industry} industry. The data is from {start_date} to {end_date}.</b>
</p>
''', unsafe_allow_html=True)

## GET DATA ##
df = psf.getStockPrices(ticker,start_date,end_date)
df = ARCHLib.calcColumns(df)

returns = df['Log_Return'].dropna()


st.markdown(
    f"<br><h2 style='font-size: 45px; text-align: left; color:black'> 2. Exploratory Data Analysis </h2>", 
    unsafe_allow_html=True
)

st.markdown(
    f"<h3 style='font-size: 35px; text-align: left; color:grey'> 2.1. Data Import & Cleaning  </h2>", 
    unsafe_allow_html=True
)

st.markdown('''
<p style='font-size:18px; text-align:left; color:black'>
API call for financial data
</p>''', unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 1, 1])  
with col1:
    st.image("EDAsnap1.png", width=1200)


st.markdown(
    f"<h3 style='font-size: 35px; text-align: left; color:grey'> 2.2. Data Overview </h2>", 
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Time Series", "Log Returns", "Summary Statistics & Stationarity","Autocorrelation","T Distribution"])

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
    results_df, interpretation = ARCHLib.adf_test_summary(df)

    st.write("## 📊 ADF Test Results (Log Returns)")
    st.dataframe(results_df)

    st.write("## 📌 Interpretation")
    st.markdown(interpretation)

with tab4:
    st.header("Autocorrelation Checks") 
    fig, ax = ARCHLib.autocorrChecks(df)
    st.pyplot(fig)

    lbResults, lbIntepretation = ARCHLib.ljung_box_test_with_interpretation(df['Log_Return'])

    st.write("## 📊 Ljung-Box Test Results (Log Returns)")
    st.dataframe(lbResults)

    st.write("## 📌 Interpretation")
    st.markdown(f'''
    <p style='font-size:18px; text-align:left; color:black'>
    {lbIntepretation}
    </p>''', unsafe_allow_html=True)

with tab5:
    ## FIT T DIST ##
    st.header("T Distribution")
    fig, statsDict = ARCHLib.fitTDist(df,companyName)
    st.pyplot(fig)
    st.dataframe(statsDict)

st.markdown(
    f"<br><br><h2 style='font-size: 45px; text-align: left; color:black'> 3. Volatility Modelling </h2>", 
    unsafe_allow_html=True
)

st.markdown(
    f"<h3 style='font-size: 35px; text-align: left; color:grey'> 3.1. Model Testing </h2>", 
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(['ARCH', 'GARCH', 'GJR-GARCH', 'EGARCH','ARCH-t', 'GARCH-t', 'GJR-GARCH-t', 'EGARCH-t'])

with tab1:
    st.subheader("ARCH Model Results")
    resultARCH, bicARCH, aicARCH, llARCH = ARCHLib.fit_arch_model(returns)
    st.write(resultARCH.summary())

with tab2:
    st.subheader("GARCH Model Results")
    resultGARCH, bicGARCH, aicGARCH, llGARCH = ARCHLib.fit_garch_model(returns)
    st.write(resultGARCH.summary())

with tab3:
    st.subheader("GJR-GARCH Model Results")
    resultGJRGARCH, bicGJRGARCH, aicGJRGARCH, llGJRGARCH = ARCHLib.fit_gjr_garch_model(returns)
    st.write(resultGJRGARCH.summary())

with tab4:
    st.subheader("EGARCH Model Results")
    resultEGARCH, bicEGARCH, aicEGARCH, llEGARCH = ARCHLib.fit_egarch_model(returns)
    st.write(resultEGARCH.summary())

with tab5:
    st.subheader("ARCH-t Model Results")
    resultARCHt, bicARCHt, aicARCHt, llARCHt = ARCHLib.fit_arch_t_model(returns)
    st.write(resultARCHt.summary())

with tab6:
    st.subheader("GARCH-t Model Results")
    resultGARCHt, bicGARCHt, aicGARCHt, llGARCHt = ARCHLib.fit_garch_t_model(returns)
    st.write(resultGARCHt.summary())

with tab7:
    st.subheader("GJR-GARCH-t Model Results")
    resultGJRGARCHt, bicGJRGARCHt, aicGJRGARCHt, llGJRGARCHt = ARCHLib.fit_gjr_garch_t_model(returns)
    st.write(resultGJRGARCHt.summary())

with tab8:
    st.subheader("EGARCH-t Model Results")
    resultEGARCHt, bicEGARCHt, aicEGARCHt, llEGARCHt = ARCHLib.fit_egarch_t_model(returns)
    st.write(resultEGARCHt.summary())


st.markdown(
    f"<br><h3 style='font-size: 35px; text-align: left; color:grey'> 3.2. Model Selection </h2>", 
    unsafe_allow_html=True
)


comparison_df = pd.DataFrame({
    "Model": ["ARCH", "GARCH", "GJR-GARCH", "EGARCH",
              "ARCH-t", "GARCH-t", "GJR-GARCH-t", "EGARCH-t"],
    "AIC": [aicARCH, aicGARCH, aicGJRGARCH, aicEGARCH,
            aicARCHt, aicGARCHt, aicGJRGARCHt, aicEGARCHt],
    "BIC": [bicARCH, bicGARCH, bicGJRGARCH, bicEGARCH,
            bicARCHt, bicGARCHt, bicGJRGARCHt, bicEGARCHt],
    "Log-Likelihood": [llARCH, llGARCH, llGJRGARCH, llEGARCH,
                       llARCHt, llGARCHt, llGJRGARCHt, llEGARCHt]
}).set_index("Model")

lowestAIC = comparison_df["AIC"].idxmin()
lowestBIC = comparison_df["BIC"].idxmin()
highestLL = comparison_df["Log-Likelihood"].idxmax()

message, best_model = ARCHLib.modelChoice(comparison_df)

st.markdown(f'''
<p style='font-size:20px; text-align:left; color:green'>
<b> 
{message}<b> 
</p>''', unsafe_allow_html=True)


st.markdown(
    f"<h2 style='font-size: 45px; text-align: left; color:black'> <br>4. Risk Modelling </h2>", 
    unsafe_allow_html=True
)

st.markdown(f'''
<p style='font-size:20px; text-align:left; color:black'>
We start by taking <b>{best_model}</b> as the best model from the analysis above.
</p>''', unsafe_allow_html=True)

if best_model == "ARCH":
    bestResult = resultARCH
elif best_model == "GARCH":
    bestResult = resultGARCH
elif best_model == "GJR-GARCH":
    bestResult = resultGJRGARCH
else:
    bestResult = resultEGARCH

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Residuals vs C.V.","Volatility","Residual Analysis","VaR Analysis","Expected Shortfall","Dynamic Risk Modelling"])

with tab1:
    col1, col2, col3 = st.columns([1, 3, 1])  
    with col2:
        st.subheader(f"{best_model} - Residuals vs. Conditional Volatility")
        st.pyplot(ARCHLib.vizBestModel(df,bestResult))

with tab2:
    col1, col2, col3 = st.columns([1, 3, 1])  
    with col2:    
        st.subheader(f"{best_model} - Volatility")
        fig, ax = ARCHLib.visVolatility(df, returns, bestResult)
        st.pyplot(fig)

with tab3:
    col1, col2, col3 = st.columns([1, 3, 1])  
    with col2:
        st.subheader(f"{best_model} - Residual Analysis")
        fig, ax = ARCHLib.residualAnalysis(bestResult)
        st.pyplot(fig)

with tab4:
    col1, col2, col3 = st.columns([1, 3, 1])  
    with col2:
        st.subheader(f"{best_model} -  VaR Analysis")
        fig, confidence_levels = ARCHLib.VaR(returns,companyName)
        st.pyplot(fig)

with tab5:
    col1, col2, col3 = st.columns([1, 3, 1])  
    with col2:
        st.subheader(f"{best_model} - Expected Shortfall")
        fig = ARCHLib.expectedShortfall(confidence_levels,companyName,returns)
        st.pyplot(fig)

with tab6:
    col1, col2, col3 = st.columns([1, 3, 1])  
    with col2:
        st.subheader(f"{best_model} - Dynamic Risk Modelling")
        fig = ARCHLib.dynamicRM(bestResult,companyName,returns)
        st.pyplot(fig)



st.markdown(
    f"<br><h2 style='font-size: 45px; text-align: left; color:black'> 5. Conclusions </h2>", 
    unsafe_allow_html=True
)






