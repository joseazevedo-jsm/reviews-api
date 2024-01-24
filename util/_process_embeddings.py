from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from umap import UMAP

model = SentenceTransformer('all-MiniLM-L6-v2')

def process_embeddings(df):
    # Initialize the model
    df = df.copy()

    # Compute embeddings
    df['embeddings'] = df['text'].apply(lambda x: model.encode(x))

    # Convert list of embeddings into a 2D array
    embeddings = np.array(df['embeddings'].to_list())

    # Use PCA to reduce the dimensionality to 2 dimensions
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    df['embeddings_2dims'] = list(embeddings_pca)

    # Apply UMAP instead... (maybe in future)
    # reducer = UMAP(n_components=2, random_state=42)
    # embeddings_umap = reducer.fit_transform(embeddings)

    # df['embeddings_umap'] = list(embeddings_umap)

    return df

   