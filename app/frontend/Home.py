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
import requests
from io import BytesIO
import json
import gc

st.set_page_config(page_title="Multimodal AI Search", layout = "wide")

# Load Drive image links
with open("image_link_mapping.json", "r") as f:
    image_link_mapping = json.load(f)

if "search_history" not in st.session_state:
    st.session_state["search_history"] = []

# Load the CLIP model once
@st.cache_resource
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    model = model.to(device)
    return model, tokenizer, device, preprocess

model, tokenizer, device, preprocess = load_clip_model()

# Helper function for loading images from URL or local path
def get_image_or_placeholder(image_path, image_url=None, caption=None):
    # 1. Try loading from URL if available
    if image_url and image_url.strip():
        try:
            response = requests.get(image_url, timeout=5) # Add timeout to prevent hanging
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img
        except Exception as e:
            pass # Fall through to local load

    # 2. Try loading from local file
    full_path = os.path.join("images", image_path)
    try:
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        
        img = Image.open(full_path).convert("RGB")
        return img
    except Exception as e:
        # 3. Return placeholder
        placeholder = Image.new('RGB', (300, 300), color=(200, 200, 200))
        return placeholder

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
    if not os.path.exists("embeddings/image_faiss.index"):
        return None, None
    index = faiss.read_index("embeddings/image_faiss.index")
    with open("embeddings/image_index_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    return index, mapping

image_faiss_index, image_mapping = load_image_faiss_index()

# Remove cache for BLIP so we can garbage collect it immediately
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    model = model.to(device)
    
    # Skip dynamic quantization as it causes OOM during conversion on 1GB instances
    return processor, model, device

# Page Setup
st.title("Multimodal AI Search Engine")

with st.sidebar:
    st.title("Smart AI Search Engine")
    st.markdown("""
Welcome to this multimodal AI search tool.

**Features:**
- Text → Image search
- Image → Caption + Similar Images
- Real-time Caption Generation (BLIP)
- Semantic Embedding Space Visualization
    """)

    st.markdown("---")
    st.markdown("**Search History**")

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
            st.markdown("### Search Results (Text Query):")

            cols = st.columns(2)  # two columns side by side

            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                with cols[i % 2]:
                    caption = mapping["captions"][idx]
                    image_filename = mapping["image_filenames"][idx]
                    image_url = mapping["image_urls"][idx]
                    
                    similarity = dist * 100
                    
                    image = get_image_or_placeholder(image_filename, image_url)
                    st.image(image, width=300, caption=f"Rank #{i+1} ({similarity:.2f}%): {caption}")

            st.session_state["search_history"].append(f"Text: {query}")


if search_mode in ["Image", "Both"]:

    st.markdown("### Image-Based Search:")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        # --- BLIP Caption Generation ---
        # 1. Load BLIP (Spikes RAM)
        blip_processor, blip_model, blip_device = load_blip_model()

        blip_image = Image.open(uploaded_image).convert("RGB")
        inputs = blip_processor(blip_image, return_tensors="pt").to(blip_device)

        with torch.no_grad():
            out = blip_model.generate(**inputs)

        generated_caption = blip_processor.decode(out[0], skip_special_tokens=True)
        
        # 2. Unload BLIP explicitly to free RAM for CLIP
        del blip_model
        del blip_processor
        gc.collect()

        # Display the caption
        st.markdown("### AI-Generated Caption (BLIP):")
        st.success(generated_caption)

        if image_faiss_index is not None:
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
                image_filename = image_mapping["image_filenames"][idx]
                image_url = image_mapping["image_urls"][idx]

                similarity = dist * 100

                image = get_image_or_placeholder(image_filename, image_url)
                st.image(image, width=300, caption=f"Rank #{rank+1} ({similarity:.2f}%)")
            
            st.session_state["search_history"].append(f"Image uploaded search – {generated_caption}")
        else:
            st.warning("Image search index not found. Run `utils/build_image_faiss_index.py` to generate it.")

