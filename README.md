# Multimodal AI Search Engine

## Overview
This project is a local AI-powered search engine that allows users to find relevant images or captions by entering a text query. It leverages OpenCLIP for embedding text into a vector space and FAISS for efficient semantic similarity search. By combining these tools, the search engine can provide meaningful results without requiring external APIs, ensuring a fast, fully offline experience.

## Features
- **Text-to-Image Search:** Enter a text query to find matching images and captions.
- **Semantic Search Backend:** OpenCLIP embeddings and FAISS vector indexing ensure high-quality, meaningful results.
- **Local Execution:** No internet connection or API keys needed — everything runs on your Mac.
- **Similarity Scores:** Results include similarity percentages to show how closely each result matches your query.

## How It Works
1. **User Query Input:** A simple Streamlit UI lets users type in a text query.
2. **CLIP Text Embedding:** The query is tokenized and converted into a high-dimensional vector.
3. **FAISS Index Search:** The vector is matched against a pre-built index of image-caption embeddings to find the most similar results.
4. **Results Display:** The top matches are displayed with their associated captions, images, and similarity scores.

## Installation
1. **Set Up Environment:**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

   deactivate
   ```

2. **Download Dataset:**
   Place images and captions in the respective `images/` and `captions/` folders. Use a formatted caption file in the form of `filename|caption`.

3. **Generate Embeddings:**
   Run:
   ```
   python3 utils/generate_clip_embeddings.py
   ```

4. **Build FAISS Index:**
   ```
   python3 utils/build_faiss_index.py
   ```

5. **Run the UI:**
   ```
   streamlit run app/frontend/search_app.py
   ```

## Project Structure
```
multimodal-search/
├── images/                   # Images dataset
├── captions/                 # Captions dataset
├── embeddings/               # Precomputed embeddings and FAISS index
├── app/
│   ├── backend/              # Backend logic (future expansions)
│   └── frontend/             # Streamlit UI
├── utils/                    # Utility scripts
└── models/                   # Pretrained model files (if needed)
```

## Usage
1. Open the UI by running the Streamlit app.
2. Enter a text query in the search box.
3. View ranked results with captions, images, and similarity scores.

## Next Steps
- **Add Image-to-Text Search:** Enable users to upload an image and retrieve relevant captions.
- **Improve Similarity Scoring:** Experiment with cosine similarity or other metrics.
- **Advanced Features:** Incorporate filtering, result explanations, and additional datasets.

---

Developed with OpenCLIP, FAISS, and Streamlit. Fully local, fast, and free to use on your Mac.