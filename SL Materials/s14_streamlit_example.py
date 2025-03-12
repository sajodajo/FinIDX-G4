import json
import zipfile
import geopandas

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px


@st.cache_data
def read_and_preprocess_data():
    with zipfile.ZipFile("data/uber-data.zip") as zip:
        with zip.open(
            "madrid-barrios-2020-1-All-DatesByHourBucketsAggregate.csv"
        ) as csv:
            data = pd.read_csv(csv)
        with zip.open("madrid_barrios.json") as geojson:
            codes = geopandas.read_file(geojson, encoding="utf-8")

    # change data types in codes because they are not the same as in data
    codes["GEOCODIGO"] = codes["GEOCODIGO"].astype(int)
    codes["MOVEMENT_ID"] = codes["MOVEMENT_ID"].astype(int)

    codes["DISPLAY_NAME"] = codes["DISPLAY_NAME"].str.split().str[1:].str.join(" ")

    # Merge the data with the codes (source)
    data = data.merge(
        codes[["GEOCODIGO", "MOVEMENT_ID", "DISPLAY_NAME"]],
        left_on="sourceid",
        right_on="MOVEMENT_ID",
        how="left",
    )
    data = data.rename(
        columns={"GEOCODIGO": "src_neigh_code", "DISPLAY_NAME": "src_neigh_name"}
    ).drop(columns=["MOVEMENT_ID"])

    data = data.merge(
        codes[["GEOCODIGO", "MOVEMENT_ID", "DISPLAY_NAME"]],
        left_on="dstid",
        right_on="MOVEMENT_ID",
        how="left",
    )
    data = data.rename(
        columns={"GEOCODIGO": "dst_neigh_code", "DISPLAY_NAME": "dst_neigh_name"}
    ).drop(columns=["MOVEMENT_ID"])

    # Create a new date column
    data["year"] = "2020"
    data["date"] = pd.to_datetime(
        data["day"].astype(str)
        + data["month"].astype(str)
        + data["year"].astype(str)
        + ":"
        + data["start_hour"].astype(str),
        format="%d%m%Y:%H",
    )

    # Create a new day_period column
    data["day_period"] = data.start_hour.astype(str) + "-" + data.end_hour.astype(str)
    data["day_of_week"] = data.date.dt.weekday
    data["day_of_week_str"] = data.date.dt.day_name()

    return data, codes


def main():
    data, codes = read_and_preprocess_data()

    sources = data.src_neigh_name.unique()
    destinations = data.dst_neigh_name.unique()
    SOURCE = st.sidebar.selectbox("Select Source", sources)
    DESTINATION = st.sidebar.selectbox("Select Destination", destinations)

    aux = data[(data.src_neigh_name == SOURCE) & (data.dst_neigh_name == DESTINATION)]
    aux = aux.sort_values("date")

    ## PLOT FIGURE 1 ##
    fig1 = px.line(
        aux,
        x="date",
        y="mean_travel_time",
        text="day_period",
        error_y="standard_deviation_travel_time",
        title="Travel time from {} to {}".format(SOURCE, DESTINATION),
        template="none",
    )

    fig1.update_xaxes(title="Date")
    fig1.update_yaxes(title="Avg. travel time (seconds)")
    fig1.update_traces(
        mode="lines+markers",
        marker_size=10,
        line_width=3,
        error_y_color="gray",
        error_y_thickness=1,
        error_y_width=10,
    )

    st.plotly_chart(fig1, use_container_width=True)

    ## PLOT FIGURE 2 ##
    aux2 = (
        aux.groupby(["day_of_week_str", "day_of_week", "day_period"])[
            ["start_hour", "mean_travel_time"]
        ]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        aux2.sort_values(["start_hour", "day_of_week"]),
        x="day_of_week_str",
        y="mean_travel_time",
        color="day_period",
        barmode="group",
        opacity=0.7,
        color_discrete_sequence=px.colors.sequential.RdBu_r,
        category_orders={
            "day_of_week_str": [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        },
        title="Avg. travel time from {} to {} by day of week".format(
            SOURCE, DESTINATION
        ),
        template="none",
    )

    fig2.update_xaxes(title="Period of Day")
    fig2.update_yaxes(title="Avg. travel time (seconds)")
    fig2.update_layout(legend_title="Day Period")

    st.plotly_chart(fig2, use_container_width=True)

    ## PLOT FIGURE 3 ##
    aux = data.groupby(["src_neigh_name"])["mean_travel_time"].mean().reset_index()
    aux = aux.set_index("src_neigh_name").join(
        codes.set_index("DISPLAY_NAME"), how="left"
    )
    aux = geopandas.GeoDataFrame.from_dict(aux.to_dict())
    color = aux.apply(
        lambda x: 1 if x.name == SOURCE else 2 if x.name == DESTINATION else 0, axis=1
    )

    fig3 = px.choropleth(
        aux,
        geojson=aux.geometry,
        locations=aux.index,
        color=color.astype(str),
        projection="mercator",
        color_discrete_sequence=["#f8f8ff", "#800000", "#808000"],
        labels=["Source"],
        title=f"Location of Source ({SOURCE}) and Destination ({DESTINATION})",
        height=600,
        width=800,
    )
    fig3.update_geos(fitbounds="locations", visible=False)
    fig3.update_layout(showlegend=False, margin={"r": 0, "l": 0, "b": 0})

    st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    # This is to configure some aspects of the app
    st.set_page_config(
        layout="wide", page_title="Madrid Mobility Dashboard", page_icon=":car:"
    )

    # Write titles in the main frame and the side bar
    st.title("Madrid Mobility Dashboard")
    st.sidebar.title("Options")

    # Call main function
    main()
