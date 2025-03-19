import faiss  # Vector search engine
import pickle
import numpy as np
import json

# Load text embeddings
with open("embeddings/text_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
image_filenames = data["image_filenames"]
captions = data["captions"]

# Load Drive image links
with open("image_link_mapping.json", "r") as f:
    image_link_mapping = json.load(f)

# Create image_urls list in same order as image_filenames
image_urls = [image_link_mapping.get(fname, "") for fname in image_filenames]

# Build FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(np.array(embeddings).astype("float32"))

# Save index
faiss.write_index(index, "embeddings/faiss_index.index")

# Save mapping
mapping = {
    "image_filenames": image_filenames,
    "captions": captions,
    "image_urls": image_urls
}

with open("embeddings/index_mapping.pkl", "wb") as f:
    pickle.dump(mapping, f)

print("✅ Text FAISS index and mapping with image URLs saved.")
