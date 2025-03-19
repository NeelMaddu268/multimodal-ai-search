import csv
import pickle

# Load your current mapping
with open("embeddings/index_mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

# Build filename → URL mapping from CSV (downloaded from Google Sheet)
link_map = {}
with open("drive_links.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row["Filename"].strip()
        link = row["Public Link"].strip()
        link_map[filename] = link

# Add image_urls list to mapping (order matches image_filenames)
image_urls = [link_map.get(fname, "") for fname in mapping["image_filenames"]]
mapping["image_urls"] = image_urls

# Save the updated mapping
with open("embeddings/index_mapping.pkl", "wb") as f:
    pickle.dump(mapping, f)

print("✅ Updated mapping file with Google Drive image URLs.")
