import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
from sklearn.preprocessing import normalize
import numpy as np
import pickle
from tqdm import tqdm
import gzip

class TextDataset(Dataset):
    def __init__(self, captions_file):
        self.captions = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or "image|caption" in line:
                    continue
                try:
                    image, caption = line.split("|", 1)
                    caption = caption.strip()
                    # Strict Filter: Must be > 15 chars AND end with punctuation
                    if len(caption) < 15 or not caption.endswith(('.', '!', '?')):
                        continue
                    if "I have no idea" in caption or "group of" == caption:
                        continue
                    self.captions.append((image, caption))
                except ValueError:
                    continue

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_name, caption = self.captions[idx]
        return caption, image_name

def collate_fn(batch):
    captions, image_names = zip(*batch)
    return list(captions), list(image_names)

def main():
    # Load CLIP model
    print("Loading CLIP model...")
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Dataset and DataLoader
    captions_file = "captions/formatted_all_captions.txt"
    dataset = TextDataset(captions_file)
    batch_size = 1000
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_embeddings = []
    
    print(f"Processing {len(dataset)} captions...")
    
    for captions_batch, _ in tqdm(dataloader, desc="Generating CLIP text embeddings"):
        tokenized = tokenizer(captions_batch).to(device)

        with torch.no_grad():
            batch_embeddings = model.encode_text(tokenized).cpu().numpy()

        all_embeddings.append(batch_embeddings)

    # Concatenate all batches
    text_embeddings = np.vstack(all_embeddings)
    text_embeddings = normalize(text_embeddings, axis=1)

    # Re-construct necessary lists for saving
    # We iterate the dataset again or just use the internal list since shuffle was False
    all_captions = [cap for _, cap in dataset.captions]
    all_image_names = [img for img, _ in dataset.captions]

    # Save final file
    output_path = "embeddings/text_embeddings.pkl.gz"
    print(f"Saving embeddings to {output_path}...")
    with gzip.open(output_path, "wb") as f:
        pickle.dump({
            "embeddings": text_embeddings,
            "caption_image_map": all_image_names,
            "captions": all_captions
        }, f)

    print("Text embeddings generated and saved.")

if __name__ == "__main__":
    main()
