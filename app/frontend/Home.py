print("\n\n\n\n\n\n\n\n")

# app/frontend/search_app.py

import streamlit as st
import torch
import open_clip
import faiss
import numpy as np
import pickle
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

import os
import zipfile
import requests

def download_and_extract_images():
    zip_path = "images.zip"
    extract_path = "images"
    
    if not os.path.exists(extract_path):
        st.warning("Downloading image dataset... (this may take a few minutes)")
        
        # Replace this with YOUR actual file ID
        file_id = "1LD7_tsjoaZBnCcigYVUE1Aj5YD0Wts65"
        gdrive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Download the zip file
        response = requests.get(gdrive_url)
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        # Extract it
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        st.success("Image dataset downloaded and extracted!")

# Run it once at start
download_and_extract_images()


st.set_page_config(page_title="Multimodal AI Search", layout = "wide")

if "search_history" not in st.session_state:
    st.session_state["search_history"] = []



# Load the CLIP model once
@st.cache_resource
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, device, preprocess

model, tokenizer, device, preprocess = load_clip_model()

# Load FAISS index and mapping
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("embeddings/faiss_index.index")
    with open("embeddings/index_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    return index, mapping

faiss_index, mapping = load_faiss_index()

@st.cache_resource
def load_image_faiss_index():
    index = faiss.read_index("embeddings/image_faiss.index")
    with open("embeddings/image_index_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    return index, mapping

image_faiss_index, image_mapping = load_image_faiss_index()

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device

blip_processor, blip_model, blip_device = load_blip_model()






# Page Setup
st.title("Multimodal AI Search Engine")

with st.sidebar:
    st.title("🧠 Smart AI Search Engine")
    st.markdown("""
Welcome to this multimodal AI search tool

**Features:**
- Text → Image search
- Image → Caption + Similar Images
- Real-time Caption Generation (BLIP)
- Semantic Embedding Space Visualization
    """)

    st.markdown("---")
    st.markdown("🔗 **Pages:** Use sidebar navigation to switch between modules.")

    st.markdown("---")
    st.markdown("🕘 **Search History**")

    for hist in reversed(st.session_state["search_history"][-10:]):
        st.markdown(f"- {hist}")



st.markdown("### Select Search Mode:")
search_mode = st.radio("Choose how you'd like to search:", ["Text", "Image", "Both"])


if search_mode == "Both":
    st.markdown("## Multimodal Search Mode")



if search_mode in ["Text", "Both"]:
    st.markdown("### Text-Based Search:")
    query = st.text_input("Search query", placeholder="e.g., a dog playing in the snow")

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:

            #Step 1: Tokenize and embed the query
            tokenized = tokenizer([query]).to(device)
            with torch.no_grad():
                query_embedding = model.encode_text(tokenized)

            query_embedding = query_embedding.cpu().numpy().astype("float32")
            from sklearn.preprocessing import normalize
            query_embedding = normalize(query_embedding, axis=1)


            # Step 2: Search FAISS index
            k = 5 # Number of results to show
            distances, indices = faiss_index.search(query_embedding, k)

            # Step 3: Display results
            st.markdown("### 🖼️ Search Results (Text Query):")

            cols = st.columns(2)  # two columns side by side

            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                with cols[i % 2]:
                    caption = mapping["captions"][idx]
                    image_path = mapping["image_filenames"][idx]
                    similarity = dist * 100
                    st.image(f"images/{image_path}", width=300, caption=f"Rank #{i+1} ({similarity:.2f}%): {caption}")
                        
            st.session_state["search_history"].append(f"Text: {query}")


if search_mode in ["Image", "Both"]:

    st.markdown("### Image-Based Search:")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])


    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # --- BLIP Caption Generation ---
        from PIL import Image

        blip_image = Image.open(uploaded_image).convert("RGB")
        inputs = blip_processor(blip_image, return_tensors="pt").to(blip_device)

        with torch.no_grad():
            out = blip_model.generate(**inputs)

        generated_caption = blip_processor.decode(out[0], skip_special_tokens=True)

        # Display the caption
        st.markdown("### AI-Generated Caption (BLIP):")
        st.success(generated_caption)


        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image_tensor)
        image_embedding = image_embedding.cpu().numpy().astype("float32")

        from sklearn.preprocessing import normalize
        image_embedding = normalize(image_embedding, axis=1)

        k = 5
        distances, indices = image_faiss_index.search(image_embedding, k)

        st.markdown("### Top Visually Similar Images:")
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            image_path = image_mapping["image_filenames"][idx]
            similarity = dist * 100
            st.image(f"images/{image_path}", width=300, caption=f"Rank #{rank+1} (Similarity: {similarity:.2f}%)")
        
        st.session_state["search_history"].append(f"Image uploaded search – {generated_caption}")



# if uploaded_image is not None:
#     image = Image.open(uploaded_image).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     # Preprocess image for CLIP
#     image_tensor = preprocess(image).unsqueeze(0).to(device)

#     # Encode image into a CLIP vector
#     with torch.no_grad():
#         image_embedding = model.encode_image(image_tensor)
    
#     image_embedding = image_embedding.cpu().numpy().astype("float32")

#     # Normalize for cosine similarity
#     from sklearn.preprocessing import normalize
#     image_embedding = normalize(image_embedding, axis=1)

#     k = 5 # Number of results to show
#     distances, indices = faiss_index.search(image_embedding, k)

#     # Step 3: Display results
#     st.markdown("### Top Search Results:")
#     for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
#         caption = mapping["captions"][idx]
#         image_path = mapping["image_filenames"][idx]
        
#         # Optionally: Convert L2 distance to similarity percentage
#         similarity = dist * 100
#         st.image(f"images/{image_path}", width=300, caption=f"Rank #{rank+1} (Similarity: {similarity:.2f}%): {caption}")
