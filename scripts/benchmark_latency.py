"""Measure collection size and search latency, for the current engine and the app it replaced.

Each configuration runs in its own process so thread limits apply from the start:
    mps    current engine on the Apple GPU
    cpu2   current engine on CPU capped at 2 threads, the vCPU count of a free Hugging Face Space
    old    the pre-rebuild app as deployed (Space commit 44167ce): ViT-B-32 text encoder, FAISS search
           over captions, then five sequential Google Drive downloads (each result was shown as it arrived)

Usage:
    python scripts/benchmark_latency.py                        # all configs -> portfolio-screenshots/latency.json
    python scripts/benchmark_latency.py --configs cpu2 old
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_SPACE, OLD_REVISION = "NM268/Multimodal-ai-search", "44167ce774ba3c6e115b3944644b4b077d320b75"
QUERIES = [
    "a dog catching a frisbee", "kids splashing in a fountain", "a surfer riding a big wave",
    "someone climbing a rock wall", "a crowd at a night concert", "two people hiking on a snowy mountain",
    "a man in a yellow kayak on rough water", "a little girl in a pink dress", "a brown horse in a field",
    "a skateboarder doing a trick on a rail", "a football player being tackled", "an old man sitting on a bench",
    "a bride and groom", "people riding bikes down a dirt trail", "a black and white dog running on the beach",
    "a baby in a bathtub", "a woman taking a photo", "a motorcycle racing around a corner",
    "a boy jumping into a swimming pool", "a city street at night", "a cat sleeping on a couch",
    "a soccer game in the rain", "a man playing guitar on stage", "children in school uniforms",
    "a red car parked on the street", "a snowboarder in the air", "a dog shaking off water",
    "a group of friends at a picnic", "a fisherman on a boat", "a person in a costume at a parade",
]


def summary(ms):
    ms = sorted(ms)
    pick = lambda q: ms[min(len(ms) - 1, round(q * (len(ms) - 1)))]
    return {"n": len(ms), "p50_ms": round(pick(0.5), 1), "p95_ms": round(pick(0.95), 1),
            "mean_ms": round(sum(ms) / len(ms), 1), "max_ms": round(ms[-1], 1)}


def run_new(device, blip):
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(ROOT / "app" / "frontend"))
    from engine import SearchEngine

    start = time.perf_counter()
    engine = SearchEngine(device=device)
    out = {"device": device, "torch_threads": torch.get_num_threads(),
           "engine_load_s": round(time.perf_counter() - start, 1),
           "photos": len(engine), "captions": len(engine.captions), "model": engine.model_name}

    for q in QUERIES[:3]:  # warm-up, not timed
        engine.search(text_vec=engine.encode_text(q))
    encode, rank, total = [], [], []
    for q in QUERIES:
        t0 = time.perf_counter()
        vec = engine.encode_text(q)
        t1 = time.perf_counter()
        engine.search(text_vec=vec, k=12)
        t2 = time.perf_counter()
        encode.append((t1 - t0) * 1000), rank.append((t2 - t1) * 1000), total.append((t2 - t0) * 1000)
    out["text_query"] = {"encode": summary(encode), "rank": summary(rank), "total": summary(total)}

    rng = np.random.default_rng(0)
    picks = [int(i) for i in rng.choice(len(engine), 10, replace=False)]
    similar = []
    for i in picks:
        t0 = time.perf_counter()
        engine.search(image_vec=engine.image_emb[i], k=12, exclude=i)
        similar.append((time.perf_counter() - t0) * 1000)
    out["more_like_this"] = summary(similar)

    images = [Image.open(load_thumbnail(engine, i)).convert("RGB") for i in picks]
    engine.encode_image(images[0])
    upload = []
    for img in images:
        t0 = time.perf_counter()
        engine.search(image_vec=engine.encode_image(img), k=12)
        upload.append((time.perf_counter() - t0) * 1000)
    out["image_query"] = summary(upload)

    if blip:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        name = "Salesforce/blip-image-captioning-base"
        processor, model = BlipProcessor.from_pretrained(name), BlipForConditionalGeneration.from_pretrained(name).eval()
        captions = []
        with torch.no_grad():
            model.generate(**processor(images[-1], return_tensors="pt"), max_new_tokens=30)  # warm-up, not timed
            for img in images[:5]:
                t0 = time.perf_counter()
                model.generate(**processor(img, return_tensors="pt"), max_new_tokens=30)
                captions.append((time.perf_counter() - t0) * 1000)
        out["blip_caption"] = summary(captions)
    return out


def load_thumbnail(engine, i):
    path = engine.thumbnail(i)
    if path.startswith("http"):
        import io
        import requests

        return io.BytesIO(requests.get(path, timeout=10).content)
    return path


def run_old():
    """Replay app/frontend/Home.py at Space commit 44167ce: same model, index and display loop."""
    import io
    import pickle

    import numpy as np
    import open_clip
    import requests
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    import faiss  # after torch: on macOS the two OpenMP runtimes segfault the other way round

    faiss.omp_set_num_threads(1)  # and FAISS's thread pool deadlocks next to torch's on macOS (not on Linux)

    files = {f: hf_hub_download(OLD_SPACE, f"embeddings/{f}", repo_type="space", revision=OLD_REVISION)
             for f in ("faiss_index.index", "index_mapping.pkl")}
    index = faiss.read_index(files["faiss_index.index"])
    with open(files["index_mapping.pkl"], "rb") as f:
        mapping = pickle.load(f)
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    def search(q):
        with torch.no_grad():
            emb = model.encode_text(tokenizer([q])).numpy().astype("float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return index.search(emb, 5)[1][0]

    search(QUERIES[0])  # warm-up
    encode_search, fetch, total, failed, duplicates = [], [], [], 0, 0
    for q in QUERIES:
        t0 = time.perf_counter()
        rows = search(q)
        t1 = time.perf_counter()
        for row in rows:  # the old app downloaded each result from Drive in turn, showing it as it arrived
            try:
                r = requests.get(mapping["image_urls"][row], timeout=5)
                r.raise_for_status()
                Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                failed += 1
        t2 = time.perf_counter()
        duplicates += 5 - len({mapping["image_filenames"][r] for r in rows})
        encode_search.append((t1 - t0) * 1000), fetch.append((t2 - t1) * 1000), total.append((t2 - t0) * 1000)
    return {
        "device": "cpu", "torch_threads": torch.get_num_threads(),
        "index_rows": int(index.ntotal), "photos": len(set(mapping["image_filenames"])),
        "results_per_query": 5,
        "text_query": {"encode_and_search": summary(encode_search), "drive_downloads": summary(fetch), "total": summary(total)},
        "failed_downloads": f"{failed}/{5 * len(QUERIES)}",
        "duplicate_photos_in_top5": f"{duplicates}/{5 * len(QUERIES)}",
    }


CONFIGS = {
    "mps": {"env": {}, "run": lambda a: run_new("mps", blip=False)},
    "cpu2": {"env": {v: "2" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
             "run": lambda a: run_new("cpu", blip=not a.skip_blip)},
    "old": {"env": {"OMP_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}, "run": lambda a: run_old()},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--out", default=str(ROOT / "portfolio-screenshots" / "latency.json"))
    parser.add_argument("--skip-blip", action="store_true")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:  # inside a configured subprocess: measure and print JSON
        if args.child in ("cpu2", "old"):
            import torch

            torch.set_num_threads(2)
        print(json.dumps(CONFIGS[args.child]["run"](args)))
        return

    results = {}
    if os.path.exists(args.out):  # re-running some configs keeps the others
        with open(args.out) as f:
            results = json.load(f)
    results["queries"] = QUERIES
    for name in args.configs:
        print(f"running {name}…", flush=True)
        cmd = [sys.executable, __file__, "--child", name] + (["--skip-blip"] if args.skip_blip else [])
        proc = subprocess.run(cmd, env={**os.environ, **CONFIGS[name]["env"]}, capture_output=True, text=True)
        if proc.returncode:
            results[name] = {"error": proc.stderr[-3000:]}
            print(f"{name} failed:\n{proc.stderr[-3000:]}")
            continue
        results[name] = json.loads(proc.stdout.strip().splitlines()[-1])
        print(json.dumps(results[name], indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
