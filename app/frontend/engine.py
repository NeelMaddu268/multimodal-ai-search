"""Search engine shared by the Streamlit pages: loads the model and precomputed embeddings, ranks images."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_REPO = "NM268/multimodal-ai-search-data"
THUMB_URL = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/main/thumbs/"

# Weight on visual similarity vs. caption similarity for text queries, tuned by scripts/eval_retrieval.py.
VISUAL_WEIGHT = 0.7


def find_data_dir():
    """Use a local build in data/ if there is one, otherwise fetch the published artifacts."""
    local = ROOT / "data"
    if (local / "index.json").exists():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(DATA_REPO, repo_type="dataset", allow_patterns=["*.json", "*.npy"]))


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)


@dataclass
class Result:
    image: int
    filename: str
    caption: str
    visual: float          # cosine similarity between the query and the photo
    caption_score: float   # best cosine similarity between a text query and the photo's captions (nan otherwise)


class SearchEngine:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir or find_data_dir())
        with open(self.data_dir / "index.json") as f:
            index = json.load(f)
        self.model_name = f'{index["model"]} ({index["pretrained"]})'
        self.images = index["images"]
        self.captions = index["captions"]
        self.caption_image = np.asarray(index["caption_image"])
        self.projected_captions = np.asarray(index.get("projected_captions", []), dtype=int)
        # captions are stored grouped by image, so each image owns the slice starts[i]:starts[i + 1]
        self.starts = np.searchsorted(self.caption_image, np.arange(len(self.images) + 1))
        self.image_emb = np.load(self.data_dir / "image_embeddings.npy").astype(np.float32)
        self.caption_emb = np.load(self.data_dir / "caption_embeddings.npy").astype(np.float32)
        projection = self.data_dir / "projection_2d.npy"
        self.projection = np.load(projection) if projection.exists() else None

        self.device = pick_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(index["model"], pretrained=index["pretrained"])
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(index["model"])
        self.encode_text("warm up")  # the first forward pass is slow; pay it at load time, not on the first search

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def encode_text(self, text):
        emb = self.model.encode_text(self.tokenizer([text]).to(self.device))
        return torch.nn.functional.normalize(emb.float(), dim=-1)[0].cpu().numpy()

    @torch.no_grad()
    def encode_image(self, image):
        emb = self.model.encode_image(self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device))
        return torch.nn.functional.normalize(emb.float(), dim=-1)[0].cpu().numpy()

    def thumbnail(self, i):
        local = self.data_dir / "thumbs" / self.images[i]
        return str(local) if local.exists() else THUMB_URL + self.images[i]

    def search(self, text_vec=None, image_vec=None, image_share=0.5, mode="hybrid", k=12, exclude=None):
        """Rank every image. Each signal is z-scored across the collection so they can be blended.

        text_vec   -> visual similarity and best-caption similarity (weighted by VISUAL_WEIGHT, or one of them via mode)
        image_vec  -> visual similarity to the query image
        Both       -> blended with image_share going to the image query.
        """
        score = np.zeros(len(self.images), dtype=np.float32)
        text_visual = text_captions = None

        if text_vec is not None:
            text_visual = self.image_emb @ text_vec
            text_captions = np.maximum.reduceat(self.caption_emb @ text_vec, self.starts[:-1])
            w = {"hybrid": VISUAL_WEIGHT, "visual": 1.0, "captions": 0.0}[mode]
            text_score = w * zscore(text_visual) + (1 - w) * zscore(text_captions)
            score += (1 - image_share if image_vec is not None else 1.0) * text_score

        if image_vec is not None:
            image_visual = self.image_emb @ image_vec
            score += (image_share if text_vec is not None else 1.0) * zscore(image_visual)

        if exclude is not None:
            score[exclude] = -np.inf

        top = np.argpartition(-score, k)[:k]
        top = top[np.argsort(-score[top])]
        visual = text_visual if text_visual is not None else image_visual
        return [self.result(i, text_vec, visual[i], text_captions[i] if text_captions is not None else np.nan) for i in top]

    def result(self, i, text_vec=None, visual=np.nan, caption_score=np.nan):
        """Package image i, labelled with the caption closest to the text query (or its first caption)."""
        lo, hi = self.starts[i], self.starts[i + 1]
        best = lo if text_vec is None else lo + int(np.argmax(self.caption_emb[lo:hi] @ text_vec))
        return Result(int(i), self.images[i], self.captions[best], float(visual), float(caption_score))

    def project(self, text_vec, k=10):
        """Place a text query on the 2-D map: similarity-weighted mean of its nearest projected captions."""
        sims = self.caption_emb[self.projected_captions] @ text_vec
        near = np.argpartition(-sims, k)[:k]
        weights = np.exp((sims[near] - sims[near].max()) * 20)
        offset = len(self.images)  # projection rows: images first, then one caption per image
        return (self.projection[offset + near] * weights[:, None]).sum(axis=0) / weights.sum()
