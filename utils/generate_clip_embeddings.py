import torch
import open_clip
from sklearn.preprocessing import normalize
import numpy as np
import pickle
from tqdm import tqdm

# Load CLIP model
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer("ViT-B-32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load captions
captions_file = "captions/formatted_all_captions.txt"
captions = []
with open(captions_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        image, caption = line.split("|", 1)
        captions.append((image, caption))

# Batch processing
batch_size = 1000
all_embeddings = []
for i in tqdm(range(0, len(captions), batch_size), desc="Generating CLIP text embeddings"):
    batch = captions[i:i+batch_size]
    text_batch = [cap for _, cap in batch]
    tokenized = tokenizer(text_batch).to(device)

    with torch.no_grad():
        batch_embeddings = model.encode_text(tokenized).cpu().numpy()

    all_embeddings.append(batch_embeddings)

# Concatenate all batches
text_embeddings = np.vstack(all_embeddings)
text_embeddings = normalize(text_embeddings, axis=1)

# Save final file (gzip compressed if preferred)
import gzip
with gzip.open("embeddings/text_embeddings.pkl.gz", "wb") as f:
    pickle.dump({
        "embeddings": text_embeddings,
        "caption_image_map": [img for img, _ in captions],
        "captions": [cap for _, cap in captions]
    }, f)

print("✅ Text embeddings generated and saved.")
