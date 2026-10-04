"""Offline pipeline: embed the Flickr8k images and captions and write everything the app loads.

Usage:
    python scripts/build_index.py --images /path/to/Flickr8k/Images
    python scripts/build_index.py --images ... --model ViT-B-32 --pretrained laion2b_s34b_b79k --out /tmp/b32 --no-thumbs

Outputs (in --out, default data/):
    index.json               model name, image filenames, captions and caption -> image index
    image_embeddings.npy     float16, L2-normalised, one row per image
    caption_embeddings.npy   float16, L2-normalised, one row per caption
    projection_2d.npy        2-D UMAP coordinates for images then one caption per image
    thumbs/*.jpg             400px thumbnails served to the browser
"""

import argparse
import contextlib
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "frontend"))
from engine import to_srgb  # noqa: E402  same colour handling as query images

THUMB_SIZE = 400
JUNK_CAPTIONS = {"a", "a group of", "broken image", "i have no idea!"}  # the only non-descriptions in Flickr8k


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def half_precision(device):
    return torch.autocast(device, dtype=torch.float16) if device != "cpu" else contextlib.nullcontext()


def clean_caption(text):
    """Undo Flickr8k's tokenised spacing: "a horse 's lead ." -> "A horse's lead.", '" sale "' -> '"sale"'."""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+([.,!?;:)]|'s\b|'S\b|n't\b)", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text)
    text = re.sub(r"(^|\s)'\s+([^']+?)\s+'(?=[\s.,!?]|$)", r"\1'\2'", text)
    return text[:1].upper() + text[1:]


def load_captions(path, image_dir):
    """Returns (image filenames, captions, caption -> image index), keeping only images present on disk."""
    on_disk = set(os.listdir(image_dir))
    by_image = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header: image,caption
        for image, caption in reader:
            caption = clean_caption(caption)
            if image in on_disk and caption.lower() not in JUNK_CAPTIONS:
                by_image.setdefault(image, []).append(caption)

    images = sorted(by_image)
    captions, caption_image = [], []
    for i, image in enumerate(images):
        for caption in dict.fromkeys(by_image[image]):  # drop exact duplicates, keep order
            captions.append(caption)
            caption_image.append(i)
    return images, captions, caption_image


@torch.no_grad()
def embed_images(model, preprocess, image_dir, filenames, device, batch_size=64):
    def load(name):
        with Image.open(os.path.join(image_dir, name)) as img:
            return preprocess(to_srgb(img))

    out = []
    with ThreadPoolExecutor(8) as pool:  # PIL decodes outside the GIL; no worker-process startup cost
        for start in tqdm(range(0, len(filenames), batch_size), desc="images"):
            batch = torch.stack(list(pool.map(load, filenames[start:start + batch_size])))
            with half_precision(device):
                out.append(model.encode_image(batch.to(device)).float().cpu())
    return torch.nn.functional.normalize(torch.cat(out), dim=-1).numpy()


@torch.no_grad()
def embed_texts(model, tokenizer, texts, device, batch_size=512):
    out = []
    for start in tqdm(range(0, len(texts), batch_size), desc="captions"):
        tokens = tokenizer(texts[start:start + batch_size]).to(device)
        with half_precision(device):
            out.append(model.encode_text(tokens).float().cpu())
    return torch.nn.functional.normalize(torch.cat(out), dim=-1).numpy()


def make_thumbnail(src, dst):
    if os.path.exists(dst):
        return
    with Image.open(src) as img:
        img = to_srgb(img)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        img.save(dst, "JPEG", quality=82, optimize=True, progressive=True)


def project_2d(image_emb, caption_emb, caption_image):
    import umap

    first_caption = np.unique(np.asarray(caption_image), return_index=True)[1]
    points = np.vstack([image_emb, caption_emb[first_caption]]).astype(np.float32)
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(points)
    return coords.astype(np.float32), first_caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder of Flickr8k .jpg files")
    parser.add_argument("--captions", default="captions/captions.txt")
    parser.add_argument("--model", default="ViT-B-16-SigLIP2")
    parser.add_argument("--pretrained", default="webli")
    parser.add_argument("--out", default="data")
    parser.add_argument("--no-thumbs", action="store_true")
    parser.add_argument("--no-projection", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    images, captions, caption_image = load_captions(args.captions, args.images)
    print(f"{len(images)} images, {len(captions)} captions")

    device = pick_device()
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    print(f"{args.model} ({args.pretrained}) on {device}")

    image_emb = embed_images(model, preprocess, args.images, images, device)
    caption_emb = embed_texts(model, tokenizer, captions, device)
    np.save(os.path.join(args.out, "image_embeddings.npy"), image_emb.astype(np.float16))
    np.save(os.path.join(args.out, "caption_embeddings.npy"), caption_emb.astype(np.float16))

    index = {
        "model": args.model,
        "pretrained": args.pretrained,
        "images": images,
        "captions": captions,
        "caption_image": caption_image,
    }

    projection_path = os.path.join(args.out, "projection_2d.npy")
    if args.no_projection and os.path.exists(projection_path):
        os.remove(projection_path)  # a stale map would no longer line up with index.json
    if not args.no_projection:
        coords, projected_captions = project_2d(image_emb, caption_emb, caption_image)
        np.save(projection_path, coords)
        index["projected_captions"] = projected_captions.tolist()

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f)

    if not args.no_thumbs:
        thumb_dir = os.path.join(args.out, "thumbs")
        os.makedirs(thumb_dir, exist_ok=True)
        with ThreadPoolExecutor() as pool:
            jobs = [pool.submit(make_thumbnail, os.path.join(args.images, n), os.path.join(thumb_dir, n)) for n in images]
            for job in tqdm(jobs, desc="thumbnails"):
                job.result()

    print(f"Wrote {args.out}/")


if __name__ == "__main__":
    main()
