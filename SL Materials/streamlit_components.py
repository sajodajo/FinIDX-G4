import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Basic text elements in streamlit
st.title("This is my first Streamlit App")
st.header("This is a header")
st.subheader("This is a subheader")
st.text("This is a text")
st.markdown("This is a **markdown** text")
st.code("# This is code\ndata = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})")
st.success("This is a success")
st.info("This is an info")
st.warning("This is a warning")

data = px.data.iris()
st.markdown("We also can include dynamic tables with `st.dataframe`")
st.dataframe(data.head())
st.markdown("And also static tables with `st.table`")
st.table(data.head())
st.markdown("And also metrics with `st.metric` combined with `st.columns`")
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")

st.markdown("Including plotly plots with `st.plotly_chart`")
st.plotly_chart(
    px.scatter(
        data,
        x="sepal_width", y="sepal_length",
        color="species", template="none"
    )
)

st.markdown("Including images with `st.image`")
st.image("https://media.giphy.com/media/zGnnFpOB1OjMQ/giphy.gif", caption='My image')

st.markdown("Including video with `st.video`")
st.video("https://youtu.be/5-tHimysW-A")

with st.container():
    st.markdown("Including widgets with `st.button`, `st.checkbox`, `st.radio`, `st.selectbox`, `st.slider`, `st.text_input`, `st.text_area`, `st.date_input`, `st.time_input`")
    st.button("This is a button")
    st.checkbox("This is a checkbox")
    st.radio("This is a radio", ("Option 1", "Option 2"))
    st.selectbox("This is a selectbox", ("Option 1", "Option 2"))
    st.slider("This is a slider", 1, 100)
    st.text_input("This is a text input")
    st.text_area("This is a text area")
    st.date_input("This is a date input")
    st.time_input("This is a time input")

    st.markdown("Including progress bars with `st.progress`")
    my_bar = st.progress(0)
    for p in range(10):
        my_bar.progress(p + 1)

    st.markdown("Or a spinner with `st.spinner`")
    with st.spinner('Wait for it...'):
        time.sleep(1) # 1 second
    st.success('Done!')


st.markdown("Separating views in different tabs with `st.tabs`")
tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
data = np.random.randn(10, 1)

tab1.subheader("A tab with a chart")
tab1.line_chart(data)

tab2.subheader("A tab with the data")
tab2.write(data)

st.markdown("This is a form with `st.form`")
with st.form("my_form"):
    st.write("Inside the form")
    slider_val = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox")

   # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("slider", slider_val, "checkbox", checkbox_val)

# Retrieve location data
st.markdown("This is a map with `st.map`")
df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(df)



# Also, we can include those elements in the sidebar
st.sidebar.title("Including things also in a sidebar!")
st.sidebar.button("This is a sidebar button")
st.sidebar.checkbox("This is a sidebar checkbox")
st.sidebar.radio("This is a sidebar radio", ("Option 1", "Option 2"))
st.sidebar.selectbox("This is a sidebar selectbox", ("Option 1", "Option 2"))
st.sidebar.slider("This is a sidebar slider", 1, 100)
st.sidebar.text_input("This is a sidebar text input")
st.sidebar.text_area("This is a sidebar text area")
st.sidebar.date_input("This is a sidebar date input")
st.sidebar.time_input("This is a sidebar time input")