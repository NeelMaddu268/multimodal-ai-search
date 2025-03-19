# app/frontend/visualize_space.py

import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

import joblib
umap_model = joblib.load("embeddings/umap_model.pkl")


with open("embeddings/embedding_projection_2d.pkl", "rb") as f:
    data = pickle.load(f)

coordinates = data["coordinates"]
labels = data["labels"]
metadata = data["metadata"]

# Build DataFrame for Plotly
df = pd.DataFrame({
    "x": coordinates[:, 0],
    "y": coordinates[:, 1],
    "label": labels,
    "meta": metadata
})


st.set_page_config(page_title="Embedding Space", layout="wide")
st.title("Visualizing CLIP Embedding Space (UMAP)")

st.markdown("### Optional: Add a Text Query Vector to the Plot")

query_text = st.text_input("Enter a query caption (optional):")

import open_clip
import torch
from sklearn.preprocessing import normalize

if query_text.strip():
    # Load CLIP model + tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Encode query caption
    tokenized = tokenizer([query_text]).to(device)
    with torch.no_grad():
        query_embedding = model.encode_text(tokenized)
    query_embedding = normalize(query_embedding.cpu().numpy(), axis=1)

    # Project into UMAP space
    query_2d = umap_model.transform(query_embedding)

    # Add query point to df
    df = pd.concat([
        df,
        pd.DataFrame({
            "x": [query_2d[0][0]],
            "y": [query_2d[0][1]],
            "label": ["query"],
            "meta": [f"🟢 {query_text}"]
        })
    ], ignore_index=True)


# Split data by type
caption_coords = coordinates[np.array(labels) == "caption"]
image_coords = coordinates[np.array(labels) == "image"]

# Create scatter plot
# Plot with Plotly
fig = px.scatter(
    df, x="x", y="y",
    color="label",
    hover_name="meta",
    title="CLIP Embedding Space (UMAP)",
    labels={"label": "Type"},
    color_discrete_map={"caption": "blue", "image": "red", "query": "green"},
    opacity=0.7,
    width=1000,
    height=700
)

st.plotly_chart(fig, use_container_width=True)