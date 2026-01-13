---
title: Multimodal AI Search
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: app/frontend/Home.py
pinned: false
---

# Multimodal AI Search Engine

## Overview
This project is a sophisticated AI-powered search engine that allows users to find relevant images or captions using natural language queries. It leverages **OpenCLIP** for state-of-the-art text-to-image and image-to-image semantic matching, and **FAISS** for high-performance vector retrieval.

The architecture is designed to be **Cloud-Ready**: heavyweight models run locally (or on the server) via PyTorch, while image assets are dynamically fetched from the cloud (Google Drive) to keep the deployment lightweight.

## Features
- **Text-to-Image Search**: Enter a text query (e.g., "dog jumping in water") to find semantically matching images.
- **Image-to-Image Search**: Upload an image to find visually similar images from the dataset.
- **Hybrid Cloud Architecture**: Uses a smart caching system to fetch images from Google Drive URLs, falling back to local files only if necessary.
- **High-Quality Data**: Implements strict dataset filtering (removing noise, fragments, and truncated captions) to ensure search relevance.
- **Embedding Visualization**: Visualizes the high-dimensional CLIP embedding space using UMAP 2D projections.
- **Real-time Captioning**: Uses the BLIP model to generate captions for uploaded images on the fly.

## PyTorch Architecture
This project demonstrates robust Deep Learning engineering patterns:
- **Custom Datasets**: Implements `TextDataset` and `ImageDataset` inheriting from `torch.utils.data.Dataset` for modular data loading.
- **Efficient Batching**: Uses `torch.utils.data.DataLoader` for optimized batch processing during embedding generation.
- **Device Agnostic**: Automatically selects CUDA (NVIDIA), MPS (Apple Silicon), or CPU based on hardware availability.

## Project Structure
```
multimodal-search/
├── images/                   # Local images (optional, used as fallback)
├── captions/                 # Captions dataset
├── embeddings/               # Precomputed FAISS indices and pickles
├── app/
│   └── frontend/
│       ├── Home.py                  # Main Streamlit Application
│       └── pages/Visualize_Space.py # Embedding Space Visualization
├── utils/                    # Data processing and embedding scripts
└── image_link_mapping.json   # Maps filenames to Google Drive URLs
```

## Installation
1. **Set Up Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Generate Embeddings (Offline Phase)**:
   This processes your captions/images and saves the vectors to `embeddings/`.
   ```bash
   python3 utils/generate_clip_embeddings.py
   python3 utils/generate_image_embeddings.py
   ```

3. **Build Indices**:
   Constructs the FAISS index for fast retrieval.
   ```bash
   python3 utils/build_faiss_index.py
   python3 utils/build_image_faiss_index.py
   ```

4. **Run the Application**:
   ```bash
   streamlit run app/frontend/Home.py
   ```

## Visualization
Navigate to the **Visualize Space** page in the sidebar to interact with a 2D UMAP projection of the semantic space. You can map your own queries to see where they land relative to "image" and "caption" clusters.

---
Developed with OpenCLIP, FAISS, PyTorch, and Streamlit.