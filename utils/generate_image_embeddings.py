# utils/generate_image_embeddings.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, image_folder, preprocess):
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.preprocess(image)
            return image_tensor, img_name, True
        except Exception as e:
            # Return dummy tensor and invalid flag
            return torch.zeros((3, 224, 224)), img_name, False

def collate_fn(batch):
    # Filter out failed images
    valid_batch = [item for item in batch if item[2]]
    if not valid_batch:
        return None, [], []
    
    tensors, names, _ = zip(*valid_batch)
    return torch.stack(tensors), list(names)

def main():
    # Step 1: Load CLIP model
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Step 2: Create Dataset and DataLoader
    image_folder = "images"
    dataset = ImageDataset(image_folder, preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

    embeddings = []
    valid_filenames = []

    # Step 3: Generate embeddings
    for batch_tensors, batch_names in tqdm(dataloader, desc="Generating image embeddings"):
        if batch_tensors is None:
            continue
            
        batch_tensors = batch_tensors.to(device)

        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_tensors)

        batch_embeddings = batch_embeddings.cpu().numpy()
        embeddings.extend(batch_embeddings)
        valid_filenames.extend(batch_names)

    if not embeddings:
        print("No valid embeddings generated.")
        return

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

if __name__ == "__main__":
    main()