import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

review_text_column="text"
colour_by_column="cluster_label"
coords_col = "embeddings_2dims"

def config_streamlit(df):

    temp_df = df.copy()
    temp_df[["x", "y"]] = temp_df[coords_col].to_list()

    # Create the interactive plot
    fig = px.scatter(
        temp_df,
        x="x",
        y="y",
        color=colour_by_column,
        hover_data={
            "x": False,  # hide the x-coordinate
            "y": False,  # hide the y-coordinate
            review_text_column: True,  # display the hover_text
        },
    )

    # print(fig)

    fig.update_layout(
        legend_title_text=None
        , height=500
    )

    # Customize the layout
    fig.update_traces(
        marker=dict(size=5, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers+text"),
    )

    # Hide cluster id -1 by default (noise)
    for trace in fig.data:
        if (
            trace.legendgroup == "Uncategorized"
        ):  # or trace.name == '-1' depending on your data
            trace.visible = "legendonly"

    # Remove x and y axis labels (set title to an empty string) and grid lines (set showgrid to False)
    fig.update_xaxes(title="", showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(title="", showgrid=False, zeroline=False, showticklabels=False)

    return fig

def visualize_cluster(df):   
    # Ensure each entry in coords_col has exactly 2 elements
    if any(len(coords) != 2 for coords in df[coords_col]):
        raise ValueError(
            f"Each entry in '{coords_col}' must have exactly 2 elements (x and y coordinates)"
        )

    # Configure 
    fig_clusters = config_streamlit(df)

    # Show
    st.plotly_chart(fig_clusters, use_container_width=True)