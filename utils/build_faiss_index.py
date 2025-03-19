import faiss # Vector search engine
import pickle
import numpy as np
import os


with open("embeddings/text_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
image_filenames = data["image_filenames"]
captions = data["captions"]

embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)

index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index, "embeddings/faiss_index.index")

mapping = {
    "image_filenames": image_filenames,
    "captions": captions
}
with open("embeddings/index_mapping.pkl","wb") as f:
    pickle.dump(mapping,f)

print("FAISS index and mapping saved.")
