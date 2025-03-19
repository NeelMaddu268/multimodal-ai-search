# utils/generate_image_embeddings.py

import os
import torch
import open_clip
from PIL import Image
import numpy as np
import pickle
from sklearn.preprocessing import normalize

# Step 1: Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Step 2: Get image filenames
image_folder = "images"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Step 3: Generate embeddings
embeddings = []
valid_filenames = []

for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    try:
        image = Image.open(img_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embedding = model.encode_image(image_tensor)

        image_embedding = image_embedding.cpu().numpy()
        embeddings.append(image_embedding[0])
        valid_filenames.append(img_name)
    except Exception as e:
        print(f"Skipping {img_name}: {e}")

# Step 4: Normalize all embeddings
embeddings = normalize(np.array(embeddings), axis=1)

# Step 5: Save the image embeddings
output = {
    "embeddings": embeddings,
    "image_filenames": valid_filenames
}

os.makedirs("embeddings", exist_ok=True)
with open("embeddings/image_embeddings.pkl", "wb") as f:
    pickle.dump(output, f)

print("Saved image embeddings to embeddings/image_embeddings.pkl")