import faiss
import pickle
import numpy as np
import json

# Load image embeddings
with open("embeddings/image_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
image_filenames = data["image_filenames"]

# Load Drive image links
with open("image_link_mapping.json", "r") as f:
    image_link_mapping = json.load(f)

image_urls = [image_link_mapping.get(fname, "") for fname in image_filenames]

# Build FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(np.array(embeddings).astype("float32"))

# Save index
faiss.write_index(index, "embeddings/image_faiss.index")

# Save mapping
mapping = {
    "image_filenames": image_filenames,
    "image_urls": image_urls
}

with open("embeddings/image_index_mapping.pkl", "wb") as f:
    pickle.dump(mapping, f)

print("✅ Image FAISS index and mapping with image URLs saved.")
