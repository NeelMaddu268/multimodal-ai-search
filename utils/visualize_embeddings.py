# utils/visualize_embeddings.py

import pickle

# Load caption embeddings
with open("embeddings/text_embeddings.pkl", "rb") as f:
    caption_data = pickle.load(f)

# Load image embeddings
with open("embeddings/image_embeddings.pkl", "rb") as f:
    image_data = pickle.load(f)

import numpy as np

# Get vectors

caption_vectors = caption_data["embeddings"]
image_vectors = image_data["embeddings"]

#Combine both vector sets
all_vectors = np.vstack([caption_vectors, image_vectors])

#Create labels of either caption or image
labels = ["caption"] * len(caption_vectors) + ["image"] * len(image_vectors)

metadata = caption_data["captions"] + image_data["image_filenames"]

import umap

#Reduce 512 dimensions into 2
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1,metric="cosine",random_state=42)
vectors_2d = umap_model.fit_transform(all_vectors)

import joblib
joblib.dump(umap_model, "embeddings/umap_model.pkl")


# Bundle everything
output = {
    "coordinates": vectors_2d,
    "labels": labels,
    "metadata": metadata
}
#Save the projection
with open("embeddings/embedding_projection_2d.pkl", "wb") as f:
    pickle.dump(output,f)

print("Saved 2D embedding projection to embeddings/embedding_projection_2d.pkl")
