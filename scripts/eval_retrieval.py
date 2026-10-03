"""Text -> image retrieval benchmark on Flickr8k.

Every human caption is used as a query (and held out of the caption index); a query is a hit at K
if the photo it describes is in the top K results. The hybrid weight is tuned on even-numbered
queries and every number reported is computed on the odd-numbered ones.

Usage:
    python scripts/eval_retrieval.py data
    python scripts/eval_retrieval.py /tmp/b32 data      # compare several builds
"""

import json
import sys

import numpy as np

KS = (1, 5, 10)
BATCH = 2048


def load(data_dir):
    with open(f"{data_dir}/index.json") as f:
        index = json.load(f)
    images = np.load(f"{data_dir}/image_embeddings.npy").astype(np.float32)
    captions = np.load(f"{data_dir}/caption_embeddings.npy").astype(np.float32)
    return index, images, captions


def zscore(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def scores_for(queries, query_ids, image_emb, caption_emb, caption_image, starts):
    """Per-query image scores: visual similarity, and best similarity among that image's other captions."""
    visual = queries @ image_emb.T
    text = queries @ caption_emb.T
    text[np.arange(len(query_ids)), query_ids] = -np.inf  # hold the query caption out
    caption_best = np.maximum.reduceat(text, starts, axis=1)
    # an image whose only caption was the query has no evidence left: give it the batch floor
    caption_best[~np.isfinite(caption_best)] = np.nanmin(np.where(np.isfinite(caption_best), caption_best, np.nan))
    top_rows = np.argpartition(-text, max(KS), axis=1)[:, : max(KS)]
    order = np.take_along_axis(text, top_rows, axis=1).argsort(axis=1)[:, ::-1]
    top_rows = np.take_along_axis(top_rows, order, axis=1)
    return visual, caption_best, caption_image[top_rows]


def rank_matrix(scores):
    """0-based rank of every image in each query's ordering."""
    order = np.argsort(-scores, axis=1)
    out = np.empty_like(order)
    np.put_along_axis(out, order, np.arange(scores.shape[1])[None, :], axis=1)
    return out


def ranks(scores, targets):
    target_scores = scores[np.arange(len(targets)), targets][:, None]
    return (scores > target_scores).sum(axis=1)


def evaluate(data_dir):
    index, image_emb, caption_emb = load(data_dir)
    caption_image = np.asarray(index["caption_image"])
    starts = np.flatnonzero(np.r_[True, caption_image[1:] != caption_image[:-1]])
    weights = np.round(np.arange(0, 1.01, 0.1), 2)

    results = {"old": [], "caption": [], "visual": [], "hybrid": {w: [] for w in weights}, "rrf": {w: [] for w in weights}}
    for start in range(0, len(caption_emb), BATCH):
        ids = np.arange(start, min(start + BATCH, len(caption_emb)))
        targets = caption_image[ids]
        visual, caption_best, old_rows = scores_for(caption_emb[ids], ids, image_emb, caption_emb, caption_image, starts)

        # Old app: top-K caption rows, shown as-is (the same photo can fill several slots)
        old_hit = np.stack([(old_rows[:, :k] == targets[:, None]).any(axis=1) for k in KS], axis=1)
        results["old"].append(old_hit)
        results["caption"].append(ranks(caption_best, targets))
        results["visual"].append(ranks(visual, targets))
        zv, zc = zscore(visual), zscore(caption_best)
        rv, rc = 1.0 / (60 + rank_matrix(visual)), 1.0 / (60 + rank_matrix(caption_best))
        for w in weights:
            results["hybrid"][w].append(ranks(w * zv + (1 - w) * zc, targets))
            results["rrf"][w].append(ranks(w * rv + (1 - w) * rc, targets))

    tune = np.arange(len(caption_emb)) % 2 == 0
    test = ~tune

    def recall(rank_list):
        r = np.concatenate(rank_list)[test]
        return [float((r < k).mean()) for k in KS]

    def tuned(kind):
        all_ranks = {w: np.concatenate(v) for w, v in results[kind].items()}
        w = max(weights, key=lambda w: (all_ranks[w][tune] < 1).mean())
        return w, all_ranks[w]

    best_w, best_hybrid = tuned("hybrid")
    rrf_w, best_rrf = tuned("rrf")
    old = np.concatenate(results["old"])[test].mean(axis=0).tolist()
    return {
        "model": f'{index["model"]} ({index["pretrained"]})',
        "queries": int(test.sum()),
        "images": len(index["images"]),
        "old app (caption match, duplicates)": old,
        "caption match, deduped": recall(results["caption"]),
        "visual (cross-modal CLIP)": recall(results["visual"]),
        f"hybrid z-score (w_visual={best_w})": recall([best_hybrid]),
        f"hybrid RRF (w_visual={rrf_w})": recall([best_rrf]),
        "best_w": float(best_w),
    }


def main():
    for data_dir in sys.argv[1:] or ["data"]:
        r = evaluate(data_dir)
        print(f"\n### {r['model']}  -  {r['queries']} held-out queries over {r['images']} images\n")
        print("| method | R@1 | R@5 | R@10 |\n|---|---|---|---|")
        for name, vals in r.items():
            if isinstance(vals, list):
                print(f"| {name} | " + " | ".join(f"{v:.1%}" for v in vals) + " |")


if __name__ == "__main__":
    main()
