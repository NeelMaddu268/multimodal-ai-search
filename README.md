---
title: Multimodal AI Search
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.65.0
python_version: "3.11"
app_file: app/frontend/Home.py
pinned: false
---

# Multimodal AI Search

Search 8,091 Flickr8k photos by describing them ("kids splashing in a fountain"), by uploading an example image, or both at once ("this photo, but at night").

**Live demo:** https://huggingface.co/spaces/NM268/Multimodal-ai-search

## How it works

```
                 ┌─────────────── offline: scripts/build_index.py ───────────────┐
 8,091 photos ──►│ SigLIP 2 image encoder ──► image_embeddings.npy   (8,091 × 768)│
40,437 captions ►│ SigLIP 2 text encoder  ──► caption_embeddings.npy (40,437 × 768)│
                 │ UMAP ──► projection_2d.npy · 400px thumbnails                  │
                 └──────────────────────────────┬─────────────────────────────────┘
                                                ▼  published to a HF dataset
 query text ──► text encoder ──┬─► cosine vs. every photo     ─┐
                               └─► best cosine vs. its captions ┼─► z-score each, blend ─► top K
 query image ─► image encoder ────► cosine vs. every photo     ─┘
```

* **Hybrid ranking.** A text query is scored two ways: against the pixels (cross-modal SigLIP 2) and against the human-written captions of each photo (five for almost all of them). Each signal is z-scored across the collection and the two are blended, so neither one's scale dominates. An image query uses visual similarity, and a text + image query blends both with a slider.
* **Exact search.** At 8k photos, scoring every image is a 768-wide matrix–vector product that takes a few milliseconds in NumPy, so there is no approximate index to tune. For millions of photos you'd swap in an ANN index (FAISS IVF/HNSW) to pull candidates and keep the same blending on top.
* **Thumbnails go straight from the Hugging Face CDN to your browser**, so the server never downloads images.
* **More like this** on any result runs an image→image search using the stored embedding.
* **BLIP** captions uploaded images, and the **Embedding space** page drops your query onto a UMAP map of all photos and captions.

## Benchmark

Every human caption is used as a query and held out of the caption index. A query counts as a hit at K if the photo it describes is in the top K results. The blend weight is tuned on half the queries, and all numbers below are from the other half (20,218 queries over 8,091 photos). Reproduce with `python scripts/eval_retrieval.py data`.

| setup | R@1 | R@5 | R@10 |
|---|---|---|---|
| **Old app:** ViT-B-32, query matched against captions, the same photo can fill several slots | 41.7% | 61.9% | 70.0% |
| ViT-B-32, query matched against the photos | 38.4% | 62.8% | 72.4% |
| ViT-B-32, hybrid | 49.9% | 72.9% | 81.0% |
| MobileCLIP2-S2, hybrid | 53.8% | 77.4% | 84.6% |
| SigLIP 2 B/16, query matched against the photos | 53.6% | 76.8% | 84.4% |
| **SigLIP 2 B/16, hybrid (shipped)** | **57.5%** | **79.7%** | **86.6%** |

The caption signal gets some help here, because the held-out query is a paraphrase written for the same photo as the captions it's matched against. That's why visual-only numbers are the fairer read on how the model handles queries in your own words. Z-score blending also beat reciprocal-rank fusion by 1 to 3 points R@1 for every model.

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/frontend/Home.py
```

On first run the app downloads the published index from [`NM268/multimodal-ai-search-data`](https://huggingface.co/datasets/NM268/multimodal-ai-search-data). If `data/` exists locally, it uses that instead.

## Rebuild the index

Download Flickr8k (for example the [Kaggle copy](https://www.kaggle.com/datasets/adityajn105/flickr8k), which ships `Images/` and the same `captions.txt` as `captions/`), then:

```bash
pip install -r scripts/requirements.txt
python scripts/build_index.py --images /path/to/Flickr8k/Images     # writes data/
python scripts/eval_retrieval.py data                               # benchmark it
```

Any [open_clip](https://github.com/mlfoundations/open_clip) model works: pass `--model` and `--pretrained`.

## Project structure

```
app/frontend/
  Home.py                    search page
  pages/Visualize_Space.py   embedding-space map
  engine.py                  model loading, encoding and ranking (no Streamlit)
  resources.py               cached models shared by the pages
scripts/
  build_index.py             embeddings, thumbnails and UMAP projection
  eval_retrieval.py          Recall@K benchmark
captions/captions.txt        Flickr8k captions (5 per photo)
```
