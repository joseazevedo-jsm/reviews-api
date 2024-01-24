import pandas as pd
import numpy as np
import hdbscan
import streamlit as st
from util.openai_service import analyze_label


EXAMPLES_NUM = 5 

def summarize_cluster(texts: list):
    """
    Generates a summary label for a cluster of customer reviews.

    Args:
        _llm (ChatOpenAI): The expert summarizer model.
        review_type (str): The type of reviews to summarize ('Likes', 'Dislikes', or 'Use-case').
        texts (list): A list of customer reviews.

    Returns:
        str: The generated summary label for the cluster of reviews.
    """
    # Use a cheaper model for the map part

    summarize_one_prompt = """
       You are skilled in summarization, distilling insights from customer reviews into natural language concise labels.

        Create a brief label (3-10 words) capturing shared sentiments across the reviews. Keep it succinct and specific, avoiding excessive length or vagueness.

        The reviews are enclosed in triple backticks (```).

        The reviews pertain to Smartphone XYZ. If it appears that most reviews do not align with this, label it as "Uncategorized."

        REVIEWS:
        ```
        {reviews_text}
        ```

        LABEL:
        """


    
    stuffed_reviews_txt = "\n\n".join(
        [f"Review {(i+1)}: {txt}" for i, txt in enumerate(texts)]
    )

    summarize_one_prompt_filled = summarize_one_prompt.format(
        reviews_text=stuffed_reviews_txt
    )


    label = analyze_label(summarize_one_prompt_filled)

    return label
    

def find_closest_data_to_centroid(df):
    # Initialize the model
    df = df.copy()

    # N closesest to center
    top_N = EXAMPLES_NUM
    closest_data = {}
    unique_clusters = df["cluster_id"].unique()

    for cluster_id in unique_clusters:
        # Get the embeddings for the current cluster
        cluster_data = df[df["cluster_id"] == cluster_id]
        cluster_embeddings = np.array(cluster_data["embeddings_2dims"].tolist())

        # Compute the centroid of the current cluster
        centroid = np.mean(cluster_embeddings, axis=0)

        # Compute the distances from the centroid to each embedding in the cluster
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

        # add something to represent if every phrase already has a cluster label

        # Get the indices of the N closest embeddings to the centroid
        closest_indices = np.argsort(distances)[:top_N]

        # Extract the corresponding text from the DataFrame
        closest_texts = cluster_data.iloc[closest_indices]["text"].tolist()

        # Store the closest data
        closest_data[cluster_id] = {"indices": closest_indices, "texts": closest_texts}  #arr for prod then add a key saying if cointains or not a cluster_label

    return closest_data

def label_cluster(closest_data):
    
    progress_bar = st.progress(0)
    progress_text = st.empty()

    top_n_cluster = closest_data
    num_clusters = len(top_n_cluster)

    for i, (cluster_id, val) in enumerate(top_n_cluster.items()):
        if cluster_id == -1:
            top_n_cluster[-1]["cluster_label"] = "Uncategorized"
        else:
            top_n_cluster[cluster_id]["cluster_label"] = summarize_cluster(val["texts"])

        progress = (i + 1) / num_clusters
        progress_bar.progress(progress)
        progress_text.text(f"Naming cluster {i + 1}/{num_clusters}")

    progress_bar.empty()
    progress_text.empty()

    return top_n_cluster

def process_cluster(df):
    # Initialize the model
    df = df.copy()

    # Fit the HDBSCAN clusterer
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1)

    # Convert embeddings to numpy array
    embeddings_pca = np.array(df["embeddings_2dims"].tolist())

    # Fit the HDBSCAN clusterer
    clusterer.fit(embeddings_pca)

    # Add cluster labels to the DataFrame
    df['cluster_id'] = clusterer.labels_

    # 
    closest_data = find_closest_data_to_centroid(df)
    
    # 
    top_cluster_docs = label_cluster(closest_data)

    # 
    top_cluster_map = {
        cluster_id: data["cluster_label"] for cluster_id, data in top_cluster_docs.items()
    }

    #
    df["cluster_label"] = df["cluster_id"].map(
        top_cluster_map
    )

    return df