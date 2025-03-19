import faiss
import pickle
import numpy as np

with open("embeddings/image_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
image_filenames = data["image_filenames"]

embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)

index.add(np.array(embeddings).astype("float32"))

# Save the image FAISS index
faiss.write_index(index, "embeddings/image_faiss.index")

# Save mapping file
mapping = {
    "image_filenames": image_filenames
}

with open("embeddings/image_index_mapping.pkl", "wb") as f:
    pickle.dump(mapping, f)

print("✅ Image FAISS index and mapping saved.")
