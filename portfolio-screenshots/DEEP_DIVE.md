# Multimodal AI Search: deep dive

[Live app](https://nm268-multimodal-ai-search.hf.space) · [Hugging Face Space](https://huggingface.co/spaces/NM268/Multimodal-ai-search) · [Published index](https://huggingface.co/datasets/NM268/multimodal-ai-search-data) · [Source](https://github.com/NeelMaddu268/multimodal-ai-search)

This Streamlit app searches the 8,091 photos of Flickr8k from a text description, an example image, or both. Every photo and its human-written captions (40,437 after cleanup) are embedded once, offline, with SigLIP 2 (ViT-B/16). A text query is scored against the photos and against their captions. Each score is z-scored over the whole collection, and the two are blended 0.7 / 0.3. The search is exact, with no approximate index: one matrix-vector product over the 8,091 photo vectors and one over the 40,437 caption vectors. With 20,218 held-out Flickr8k captions as queries, the right photo comes first 57.5% of the time and lands in the top 10 86.6% of the time. A re-creation of the old app's method scores 41.7% and 70.0% on the same benchmark (the old app only ever showed 5 results). On the live Space (free tier, 2 shared vCPUs), a new text query takes 73-76 ms on the server. In a browser on a home connection, the first result's caption appears about 0.3 s after pressing Enter, and all 12 photos have loaded by about 0.6 s. The current portfolio line, "Searches 10,000+ images in under 2s with CLIP and FAISS", gets the image count wrong, names a model and an index the app no longer uses, and undersells the current app's speed. A corrected line closes the speed section.

| Headline | Value |
|---|---|
| Collection | 8,091 photos, 40,437 captions (Flickr8k) |
| Model | SigLIP 2 ViT-B/16 (open_clip `ViT-B-16-SigLIP2`, `webli`), 375,187,970 parameters |
| Recall@1 / @5 / @10 | 57.5% / 79.7% / 86.6% (old app's method, re-created: 41.7% / 61.9% / 70.0%) |
| Server time per new text query, live Space (2 vCPU) | 73-76 ms, median 74.5 ms (n=8) |
| Enter to first result on screen, live Space, home browser | 306-341 ms, median 327 ms (n=7, uncached) |
| Enter to all 12 photos loaded, live Space, home browser | 564-707 ms, median 589 ms (n=7, uncached) |
| Text query, engine only, local Apple M5 GPU | 10.5 ms median (n=30) |

## Screenshots

![Search results for "a man in a yellow kayak on rough water"](search-yellow-kayak.png)

*"a man in a yellow kayak on rough water", 12 results in 40 ms (local app on an Apple M5 MacBook, Apple GPU, warm). Results #6 and #8 are the same photo, stored under two file names in Flickr8k.*

![Search results for "two dogs playing tug of war in the snow"](search-dogs-snow.png)

*"two dogs playing tug of war in the snow", 12 results in 43 ms (same machine and settings).*

## What it does

- **Text search.** Type a description. A sidebar control picks how text is ranked: **Hybrid** (default) blends both signals, **Visual** compares the words to the pixels, and **Captions** compares them to the human captions.
- **Image search.** Upload a JPG, PNG or WebP file and the app ranks photos by visual similarity. BLIP writes a caption for the upload, which appears next to it.
- **Text + image.** Use both inputs at once. A "Text ↔ image balance" slider (0 to 1, default 0.5, step 0.05) then appears.
- **More like this.** Every result card has a button that runs an image-to-image search from that photo.
- **Result cards.** Each card shows the 400 px thumbnail (cropped to 4:3 on the card), the caption closest to the text query (or the photo's first caption when there is no text), its rank, and the raw score for each signal used (for example `#1 · visual 0.166 · caption 0.950`). The page shows 8, 12, 16, 24, 32 or 48 results (default 12).
- **Browsing.** With no query, the page shows a fixed pseudo-random set of photos (the same for every new visitor) and a Shuffle button that draws a new set. Recent searches are listed in the sidebar.
- **Embedding-space map.** A second page plots all photos and one caption per photo in 2-D with UMAP, and drops your query onto the map.

## How it works

```
offline (scripts/build_index.py)
  8,091 photos ──► SigLIP 2 image tower ──► image_embeddings.npy    8,091 × 768, fp16
 40,437 captions ► SigLIP 2 text tower  ──► caption_embeddings.npy 40,437 × 768, fp16
                   UMAP ──► projection_2d.npy       400 px JPEGs ──► thumbs/
                                       │
                                       ▼  HF dataset NM268/multimodal-ai-search-data
                 ┌─────────────────────┴─────────────────────┐
   *.json + *.npy (77.5 MB)                         thumbs/ (237.4 MB)
                 ▼                                           ▼
   HF Space, 2 vCPU: encode query, score all           Hugging Face CDN
   8,091 photos, send top k captions + image URLs            │
                 │                                           │
                 └───────────────► browser ◄─────────────────┘
```

### Offline build

`scripts/build_index.py` runs once and writes everything the app loads.

- **Captions.** `captions.txt` has 40,455 rows (8,091 photos × 5). The build undoes Flickr8k's tokenised spacing ("a horse 's lead ." becomes "A horse's lead."). It drops 5 junk captions ("a" twice, "a group of", "broken image", "i have no idea!") and 13 exact duplicates, which leaves 40,437. Of the photos, 8,073 keep 5 captions and 18 keep 4.
- **Colour.** `to_srgb()` applies EXIF rotation, converts to RGB, and converts any embedded RGB ICC profile to sRGB (one photo with an unusable CMYK profile is left as is). 4,125 of the 8,091 photos embed a profile, and 1,317 of those are not named sRGB: Adobe RGB (1998) 631, Camera RGB Profile 499, Generic RGB Profile 107, and a tail of others including ProPhoto RGB. The conversion changes pixels by a mean of more than 1/255 in 1,356 photos. Query uploads go through the same function, so uploads and indexed photos are handled the same way.
- **Embeddings.** The image tower encodes each photo and the text tower encodes each caption. Vectors are L2-normalised and stored as float16: 12.4 MB for the photos and 62.1 MB for the captions.
- **Thumbnails.** Each thumbnail has a longest side of 400 px (LANCZOS, never upscaled) and is saved as a progressive JPEG at quality 82. The 8,091 files total 237.4 MB, an average of 28.7 KiB each.
- **Map.** UMAP (`n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42`) runs on the photo vectors plus the first caption of each photo, giving 16,182 points.

The app loads four files totalling 77,544,749 bytes. Everything, thumbnails included, is published to the HF dataset above.

### Model

| | |
|---|---|
| open_clip model | `ViT-B-16-SigLIP2`, pretrained `webli` |
| Parameters | 375,187,970 in total. Image tower 92,884,224. Text tower 282,303,744, of which 196,608,000 is the token-embedding table (256,000 vocab × 768). |
| Embedding size | 768 for both images and text |
| Image input | 224 × 224 with 16 px patches (196 patches). Photos are resized straight to 224 × 224 (bicubic, "squash"), so there is no centre crop and the aspect ratio is not kept. |
| Text input | 64 tokens, Gemma tokenizer. Text is lowercased and stripped of punctuation, and longer input is truncated without warning. "A Dog's Frisbee!" and "a dogs frisbee" give identical tokens. |
| Weights | 1,500,786,832 bytes on disk (fp32). They take 1,431.2 MiB in memory. |

The text tower is about three times the size of the image tower, mostly because of its vocabulary table.

### Ranking a text query

Each photo gets two scores for a query vector `q`:

- **visual**: the cosine between `q` and the photo's embedding (`image_emb @ q`).
- **caption**: the best cosine between `q` and any of that photo's captions (`np.maximum.reduceat(caption_emb @ q, starts[:-1])`).

These two can't simply be added, because text-to-image and text-to-text cosines live on very different scales. For "a man in a yellow kayak on rough water", the visual cosine across the 8,091 photos has mean −0.079, std 0.045 and max 0.166. The caption cosine has mean 0.406, std 0.082 and max 0.964. Across the collection, a caption scores 0.157 against its own photo on average (text to image), but 0.428 against a caption of an unrelated photo (text to text). On raw scores, a wrong photo's caption beats the right photo's pixels. So each signal is standardised per query before blending:

```
z(x)  = (x − mean(x)) / (std(x) + 1e-6)          over all 8,091 photos, per query, per signal
score = w · z(visual) + (1 − w) · z(caption)      w = 0.7 Hybrid, 1.0 Visual, 0.0 Captions
```

Worked example, using the rounded statistics above: the kayak query's top result has visual 0.166 (about 5.4 standard deviations above the mean) and caption 0.950 (about 6.6). Its blended score is about 0.7 × 5.4 + 0.3 × 6.6 ≈ 5.8.

The weight `w = 0.7` is the value in {0.0, 0.1, …, 1.0} with the best Recall@1 on half the benchmark queries (see below). The top k come from `np.argpartition` followed by a sort of those k. Every photo is scored on every query. On 30 natural queries, Hybrid and Visual put a different photo first 13 times (43%), and their top 12 lists share 9.3 photos on average.

Each result card shows the caption closest to the query. That makes the label match the query as well as possible, which also has a downside (see Limits).

### Image and text + image queries

- **Image only.** The upload goes through `to_srgb()`, the model's preprocessing and the image tower, giving a vector `v`. The score is `z(image_emb @ v)`.
- **Text and image.** The score is `(1 − s) · text_score + s · z(image_emb @ v)`, where `s` is the balance slider.
- **More like this.** This reuses the photo's stored embedding, so no model runs. The source photo is excluded. A z-score of a single signal keeps its order, so the ranking is plain cosine order.

### BLIP captions for uploads

BLIP is `Salesforce/blip-image-captioning-base` (223,971,644 parameters). It loads on the first upload, not at startup, and generates up to 30 new tokens. Results are cached for 32 uploads. The caption is only displayed and plays no part in ranking. On the M5 with 2 CPU threads it takes 241.0 ms median per image (n=5).

### The map page

The page plots 16,182 points with Plotly WebGL: 8,091 photos in red and 8,091 captions in blue, one caption per photo. When you type a query, the 10 photos the hybrid search returns are drawn as gold dots. These come from the default Hybrid mode whatever the Home page's setting. The query itself is a green star. The map has no UMAP model at run time, so the star is placed by `engine.project()`. It takes the 10 projected captions nearest the query and averages their 2-D positions, weighting each by `exp(20 · (sim − max_sim))`. The star's position is therefore an estimate.

### Serving and caching

- **One engine per server process.** `get_engine()` is wrapped in `st.cache_resource`, so all sessions share one model and one copy of the embeddings. A warm-up forward pass runs at load.
- **Per-input caches.** These use `st.cache_data`: text embeddings (256 entries), decoded uploads (8), upload embeddings (32) and BLIP captions (32). A repeated text query skips the model and only re-ranks.
- **Index download.** `data/` is not in the Space repo. On first start the app downloads only `*.json` and `*.npy` (77.5 MB) from the dataset.
- **Thumbnails go straight to the browser.** The app gives the browser `https://huggingface.co/datasets/NM268/multimodal-ai-search-data/resolve/main/thumbs/<file>` URLs. These answer with a redirect to Hugging Face's CDN (CloudFront), so the Space never downloads or serves an image.
- **Hosting.** The Space runs on `cpu-basic` (2 vCPU, 16 GB RAM, free) with 1 replica, and sleeps after 48 h without traffic. It uses Streamlit 1.65.0 and Python 3.11, with every direct dependency pinned (for example torch 2.14.1, open_clip_torch 3.3.0). The Space runs the same app and scripts as the GitHub repository.

### What the rebuild replaced

The old app is the Space at commit `44167ce`.

| | Old app (`44167ce`) | Current app (`16ee765`) |
|---|---|---|
| Model | ViT-B-32 `laion2b_s34b_b79k`, 151,277,313 params, 512-d | SigLIP 2 B/16 `webli`, 375,187,970 params, 768-d |
| What a text query is compared to | Captions only (34,184 left after a filter) | Photos and captions, blended after z-scoring |
| Search | FAISS `IndexFlatIP` (also exact) with one row per caption, top 5 | Exact NumPy scoring of every photo and caption vector, best caption per photo, top 8 to 48 |
| Duplicates in results | The same photo could fill several slots (17 of 150 in the replay) | One row per photo |
| Photos reachable by text | 8,088 (the caption filter left 3 with none) | 8,091 |
| Showing results | The server downloaded each of the 5 images from Google Drive in turn and showed it as it arrived (all five after ~6 s in the replay) | Browser loads thumbnails from the HF CDN |
| BLIP | Reloaded on every rerun with an upload | Loaded once, cached |
| Text + image | Two separate result lists | One blended ranking with a slider |
| Map | Loaded a 250.7 MB UMAP pickle; placing a query was disabled if the pickle failed to load | Precomputed coordinates; query placed from its nearest captions |
| Stored artifacts | 426.7 MB of LFS embedding files | 77.5 MB index plus 237.4 MB of thumbnails |

The old caption filter kept a caption only if it was at least 15 characters long and ended in `.`, `!` or `?`. It dropped 6,271 of 40,455 captions, 6,227 of them for lacking final punctuation.

## How well it works

**Protocol** (`scripts/eval_retrieval.py`):

- Every Flickr8k caption is used as a query, and that caption is held out of the caption index while it is the query.
- A query is a hit at K if the photo it was written for is in the top K.
- The blend weight is tuned on the even-numbered queries.
- All numbers below come from the odd-numbered queries: 20,218 queries over 8,091 photos.

| Setup | R@1 | R@5 | R@10 | Re-run for this write-up |
|---|---|---|---|---|
| Old app's method, re-created: ViT-B-32, query vs captions, a photo can fill several slots | 41.7 | 61.9 | 70.0 | No¹ |
| ViT-B-32, hybrid z-score | 49.9 | 72.9 | 81.0 | No¹ |
| MobileCLIP2-S2, hybrid z-score | 53.8 | 77.4 | 84.6 | No¹ |
| SigLIP 2, query vs captions only (one row per photo) | 42.8 | 64.2 | 72.3 | Yes |
| SigLIP 2, query vs photos only (Visual) | 53.6 | 76.8 | 84.4 | Yes |
| SigLIP 2, hybrid reciprocal-rank fusion (best weight, 0.9) | 54.7 | 78.4 | 85.7 | Yes |
| **SigLIP 2, hybrid z-score, w = 0.7 (shipped)** | **57.5** | **79.7** | **86.6** | Yes |

¹ These rows come from an earlier run of the same script. The ViT-B-32 and MobileCLIP2-S2 builds are no longer on disk, so I could not re-verify them. Run on the SigLIP 2 build, the script's "old app" row prints 42.7 / 62.0 / 70.0, because that row uses whichever build it is given.

What the table shows:

- **The rebuild added 15.8 points of R@1, about half from each change.** Keeping ViT-B-32 and switching to the z-scored hybrid adds 8.2 (41.7 → 49.9). Swapping in SigLIP 2 adds another 7.6 (49.9 → 57.5). On the old caption-only method SigLIP 2 barely helps (42.7-42.8 against 41.7): its advantage is in matching text to pixels.
- **Blending helps on this benchmark.** The hybrid adds 3.9 points over visual-only. Both that gain and w = 0.7 were measured on caption-style queries, which favour the caption signal (see the caveat). On free-form queries it can hurt: for "a brown horse in a field", Hybrid shows 4 horse photos against Visual's 7 (see Limits).
- **Z-score fusion beats reciprocal-rank fusion.** It leads by 2.8 points R@1 for SigLIP 2, and by 1 to 3 points for ViT-B-32 and MobileCLIP2-S2 in the earlier run (¹).

**Caveat.** The caption signal has an advantage on this benchmark. Each query is a caption written by a Flickr8k annotator for the same photo whose other captions it is matched against, so it is a close paraphrase in the same style. Real queries usually are not. The visual-only row (53.6 / 76.8 / 84.4) doesn't rely on matching other annotators' wording, so it is less inflated. Its queries are still Flickr8k captions, though, and the app ships Hybrid. Neither row measures queries in your own words; the Limits section below covers those by example.

Two smaller leaks are unmeasured. SigLIP 2 (WebLI) and ViT-B-32 (LAION-2B) were trained on web-scraped image-text pairs that may include Flickr8k's public Flickr photos, which would inflate the visual rows, possibly by different amounts for each model. And the tune/test split is by caption, not by photo, so w = 0.7 was chosen on the same photos it is tested on (it is one parameter, so the effect is probably small).

Only text queries are scored. Image search, text + image search (including the slider's untuned 0.5 default) and More like this have no accuracy number.

## How fast it is: re-measuring the portfolio claims

The current portfolio line reads "Searches 10,000+ images in under 2s with CLIP and FAISS." Each part was checked against the live Space and the code.

### "10,000+ images": false

The collection is **8,091 photos**, all of Flickr8k, with 40,437 captions. The app's sidebar says so, and so does `data/index.json`. It has never been larger: the old app's index had 34,184 caption rows covering 8,088 photos. "CLIP and FAISS" is also out of date. The model is SigLIP 2, and the search is exact NumPy scoring with no FAISS. The old FAISS `IndexFlatIP` was exact search too, so dropping it changed neither accuracy nor speed at this size.

### "under 2s": true of the current app for warm text queries; the old app took ~6 s to show all five results

**Live Space** (`cpu-basic`, 2 shared vCPUs, 16 GB). Queries were typed into a browser on a home connection on 2026-10-03, 21:02-21:10 EDT, right after a rebuild. One person, one browser tab, no other traffic. The Space runs 1 replica on 2 vCPUs, so concurrent visitors would queue, and these are best-case times. The server time is the app's own on-screen timer: text encoding plus ranking all 8,091 photos. Browser times come from a script in the page that measured from dispatching Enter until the new #1 caption appeared, and until all 12 thumbnail images reported loaded.

| Query | Server time | Enter to first result's caption on screen | Enter to all 12 thumbnails loaded |
|---|---|---|---|
| a man in a yellow kayak on rough water | 76 ms | not timed | not timed |
| a dog catching a frisbee | 6 ms (cached) | 262 ms | 602 ms |
| a fisherman standing on a small wooden boat at dawn | 76 ms | 341 ms | 707 ms |
| three kids jumping on a trampoline in a backyard | 76 ms | 327 ms | 589 ms |
| an old woman knitting by a window | 73 ms | 329 ms | 667 ms |
| a dog catching a frisbee (repeat) | 6 ms (cached) | 236 ms | 499 ms |
| a firefighter spraying water on a burning house | 74 ms | 306 ms | 700 ms |
| two boys wrestling in the mud | 74 ms | 309 ms | 572 ms |
| a woman in a red coat walking in the rain | 73 ms | 308 ms | 569 ms |
| a cyclist racing downhill on a mountain road | 75 ms | 327 ms | 564 ms |

- **Server time:** 73-76 ms for the 8 uncached queries, median 74.5 ms. A cached query took 6 ms, because only the ranking ran.
- **Enter to first result's caption on screen:** 306-341 ms for the 7 uncached runs (median 327 ms). Thumbnails were still loading at that point.
- **Enter to all 12 thumbnails loaded from the CDN:** 564-707 ms, median 589 ms, for the 7 uncached runs. The two frisbee runs (602 and 499 ms) are left out because the browser had probably cached their photos already.

Image queries and BLIP were not timed on the Space. The first upload after a restart also has to download and load BLIP (~0.9 GB of weights), which will take well over 2 s.

**Local Apple M5 MacBook** (10 cores, 16 GiB), from `scripts/benchmark_latency.py`. Text runs use 30 queries after 3 untimed warm-ups. `mps` runs the model on the Apple GPU, and `cpu2` caps the CPU at 2 threads to approximate the Space. The p95 is a nearest-rank pick, so for n=10 and n=5 it equals the maximum.

| Step | M5 GPU (`mps`), p50 / p95 | M5 CPU, 2 threads (`cpu2`), p50 / p95 | n |
|---|---|---|---|
| Encode text | 8.3 / 8.9 ms | 19.6 / 49.3 ms | 30 |
| Rank all 8,091 photos | 2.1 / 2.6 ms | 2.4 / 4.8 ms | 30 |
| Text query, total | 10.5 / 11.0 ms | 21.9 / 53.0 ms | 30 |
| More like this (no model run) | 0.4 / 0.4 ms | 1.1 / 2.8 ms | 10 |
| Image query (encode a 400 px thumbnail, then rank) | 25.3 / 25.8 ms | 53.6 / 103.3 ms | 10 |
| BLIP caption of an upload | not run | 241.0 / 263.3 ms | 5 |
| Engine load (model already on disk) | 8.2 s | 7.9 s | 1 |

These numbers are not interchangeable:

- **Space vs local `cpu2`.** Not like for like. The Space figure (73-76 ms) is the app's on-screen timer, which includes Streamlit and caching overhead. On the M5 GPU the same timer read 40-43 ms against 10.5 ms for the bare engine loop. The Space is probably slower than 2 threads on the M5 (21.9 ms median), but these numbers can't say by how much.
- **Screenshots vs benchmark.** The two screenshots (40 ms and 43 ms, local app on the M5 GPU) come from the app's on-screen timer inside a running Streamlit server, so they read higher than the bare engine loop on the same machine (10.5 ms median).

**Old app replay.** `scripts/benchmark_latency.py --configs old` replays the old app's text search on the M5. It loads the same ViT-B-32 model and FAISS index from Space commit `44167ce`, limits the CPU to 2 threads, then does the same five sequential Google Drive downloads. The replay used 30 queries, so 150 downloads, and none failed. The ~6 s per query below was measured by this replay **from a home connection, not from the Space itself**.

| Old app step | p50 | p95 | max |
|---|---|---|---|
| Encode + FAISS search | 34.9 ms | 60.7 ms | 87.0 ms |
| 5 sequential Google Drive downloads | 5,901.8 ms | 7,253.4 ms | 7,366.9 ms |
| Total until all 5 results shown | 5,961.4 ms | 7,285.9 ms | 7,402.2 ms |

So the old "under 2s" held for the search step, but the full set of five results took about 6 s in the replay (downloads mean 6,045.3 ms; total mean 6,085.1 ms), because each image was downloaded in turn and shown as it arrived. The replay did not time the first result, which probably appeared after roughly one download (about 1.2 s). The current app sends thumbnail URLs to the browser instead. On the live Space the first caption shows in ~0.3 s and all 12 photos in ~0.6 s. The old figure is a home-connection replay, not a Space measurement, so treat the roughly 10× gap as approximate.

**Cold start.** The first page load after the rebuild showed the UI about 20 s after the request (live Space, including downloading the model weights and the index). Locally, with the model already downloaded, the engine loaded in 7.9-8.2 s in the benchmark run. The Space sleeps after 48 h without traffic. A wake from sleep was not timed separately.

**Memory** (local M5, CPU engine, macOS physical footprint, which was consistent across 3 runs):

| Process state | Footprint |
|---|---|
| numpy, torch and open_clip imported | 447 MiB |
| SearchEngine loaded | 2,321 MiB (peak during load: 2,611 MiB) |
| plus BLIP, with its weights fully resident | 3,254 MiB (about 3.2 GiB) |

Of the engine's footprint, the SigLIP 2 weights are 1,431.2 MiB and the embeddings take 142.2 MiB once upcast to float32. BLIP's weights are memory-mapped from disk, so right after loading it the footprint shows only 2,426 MiB. The 3,254 MiB figure counts them as resident. Either way it fits easily in the Space's 16 GB.

### Verdict

| Claim | Measured | Status |
|---|---|---|
| 10,000+ images | 8,091 photos, 40,437 captions | False |
| under 2 s | 73-76 ms server time; ~0.3 s to the first result on screen; ~0.6 s to all 12 photos (live Space, home browser, warm text queries) | True for warm text queries. Not on a cold start (~20 s) or the first image upload after a restart. The old app took ~6 s to show all five results (replay). |
| CLIP | SigLIP 2 ViT-B/16 | Out of date |
| FAISS | Exact NumPy scoring, no FAISS (the old FAISS index was exact search too) | Out of date |

Suggested replacement blurb:

> Searches 8,091 Flickr8k photos by text, image, or both with SigLIP 2; on a free 2-vCPU Hugging Face Space, a text search shows its first result in ~0.3 s and all 12 photos in ~0.6 s.

## Limits

These come from probing the shipped engine on the published index. Unless noted, each query used Hybrid mode with k = 12. "By eye" means I looked at the result photos, not just their captions.

**What works**

| Query | What came back |
|---|---|
| three dogs | All 12 have a caption saying three dogs (10/12 in Visual), although only 94 photos (1.2%) qualify. "two dogs": 11/12. |
| a dog with a red ball | "A dog with a red ball.", "The dog is playing with a red ball." With one object and one colour, the colour attaches to the right thing. |
| un perro corriendo en la playa | 12/12 dogs on beaches in every mode. 11 of 12 in Hybrid (9 in Visual) are the same photos as for "a dog running on the beach". The two queries' embeddings have cosine 0.932. |
| ビーチを走る犬 (Japanese, "a dog running on a beach") | 12/12 correct in every mode. 8 of 12 match the English results in Hybrid, but only 3 of 12 in Visual. |
| a dgo cathcing a frisbe | 12/12 frisbee photos in both modes. 7 (Hybrid) and 5 (Visual) of them are the same as for the correct spelling. |
| an airplane taking off | "A small jet takes off." comes first. 34 photos mention planes or jets. |
| a sign that says stop | By eye, the top 3 all show the word STOP. #2 is labelled "A female jogger wearing red.", though another of its captions mentions a stop sign. It is also the top result in Visual mode, so the image encoder found it on its own. #4-12 are generic street scenes. |
| Query image: the word "DOG" in black on white | 12/12 dog photos, where 3.0 per 12 would be expected by chance. An image of the word "snow" gives 10/12 snow photos. |

**What fails or only partly works**

| Query | What came back |
|---|---|
| five dogs running in a field | 2 of 12 show five dogs (12 such photos exist); by eye, most of the rest show four. "five dogs" alone: 1/12. Counting holds up to about three. |
| a beach with no people | By eye, only 2 of 12 have no people. The top result is the same as for "a beach with people" (text cosine 0.926). "a dog without a ball": 10/12 show a ball. |
| a man to the left of a dog | 11 of 12 photos and the same top result as "a dog to the left of a man" (text cosine 0.987). |
| a man in a red shirt and a woman in a blue shirt | Swapping the colours returns the identical 12 photos in both modes (text cosine 0.9965). By eye, the results include both colour assignments. |
| a horse riding a man | Ordinary horse-and-rider photos; 9 of 12 are shared with "a man riding a horse". |
| a cat on top of a car | No cat in the top 12; the best cat photo is 13th. "a cat" alone: 7/12. Only 23 photos contain a cat. |
| a giraffe | No caption mentions a giraffe. By eye, the results are mostly dogs, plus a deer, a bird, a net on a hillside and a child in face paint, and nothing on screen says the match is poor. Top visual score 0.081. |
| a brwon hrose in a feild | No horses at all, against 4/12 (Hybrid) and 7/12 (Visual) for the correct spelling. The results fall back to dogs. |
| a brown horse in a field | Only 4 of 12 show horses, though 84 horse photos exist. Two "A brown dog in a field." photos get in on exact caption wording, with visual scores of only 0.067-0.072. |
| A 67-word query: a 51-word beach description followed by "…but what I actually want is a black dog jumping over a fence in the snow" | The tokenizer cuts the text at 64 tokens. What it keeps ends "…what i actually want is a black". None of the 12 has a dog. With the request moved to the front: 11/12 dogs. |
| a jersey with the number 23 | Jerseys numbered 25 and 28. "a sign that says no parking" finds a parking lot. |
| 赤い車 (Japanese, "red car") | 6/12 red cars in Hybrid and 3/12 in Visual, where results drift to red clothing. English gets 8/12 in both modes. |
| Query image: flat red, blue, white or black, or random noise | The same two photos (a blue pool and an empty stadium) appear in all five top 12s. Red and blue share 7 of 12 results, so this is not colour search. |
| Query image: the kayak query's top photo, mirrored, cropped 10% on each side, greyscale | All 12 results are kayak photos, but the original photo ranks only 9th (0.840). |

**Patterns behind these**

- **Word order, negation and relations barely move the query.** Swapped or negated queries have text embeddings with cosine 0.913 to 0.9965, so the search mostly sees which things are mentioned, not how they relate. "no" is effectively ignored. The swapped pairs below share most of their top 12 (Hybrid / Visual):

  | Pair | Shared of 12 |
  |---|---|
  | red/blue shirts | 12 / 12 |
  | black dog chasing white dog, swapped | 11 / 11 |
  | man left of dog vs dog left of man | 11 / 10 |
  | man riding horse vs horse riding man | 9 / 11 |
  | dog on car vs car on dog | 6 / 7 |
- **Nothing tells you when a match is poor.** The app always fills k slots. For text, the top visual score is a weak hint at best. Queries for things the dataset lacks scored 0.077-0.098 (for example "a giraffe" 0.081), and rare things that are present scored 0.113-0.137 (for example "an airplane taking off" 0.113). But "a cat sleeping on a couch" scored 0.077 with real cats in its top 3, and "a laptop on a desk" scored 0.078 and still found laptops. For image queries the gap is clearer. Synthetic images topped out at 0.58-0.67, while real photos' nearest neighbours have a 5th percentile of 0.753.
- **The data is skewed.** 25.0% of photos have a dog in a caption and 73.4% mention a man, woman, boy, girl, child or people. Cats are in 0.3% of photos, horses in 1.0% and cars in 2.2%. When a query is ambiguous or misspelled, rarer subjects lose to dogs.
- **Long queries are cut off silently** at 64 tokens, and the UI says nothing. In the test above, 59 of the 67 words survived. Plain wording fits more: a 50-word plain-English description used 53 tokens and was not cut.
- **Captions on result cards can flatter the results.** Each card shows whichever of the photo's captions best matches the query, so the label is picked to agree with the query and a weak result can still look on-topic. The chosen label is not always the most relevant caption either. For "a sign that says stop", #2 is labelled "A female jogger wearing red." (cosine 0.610), even though another of its captions, "The lady in red shorts jogs near a stop sign while listening to music.", is the obvious match (0.579).
- **Duplicates are not collapsed.** Flickr8k contains one byte-identical pair (`2851198725_37b6027625.jpg` and `3050606344_af711c726c.jpg`): results #6 and #8 in the kayak screenshot. Each copy has its own captions, so the same pixels get different caption scores (0.856 and 0.806) and rank two places apart. 86 photo pairs have image cosine above 0.95, mostly burst shots.
- **The image encoder reads large text.** That helped with STOP, but it also means a picture of a word steers image search toward that word. It does not reliably match specific strings such as numbers.
- **Photos are squashed to a square** before encoding. I did not measure the effect on accuracy.
- **Accuracy outside Flickr8k-style captions is unscored.** The benchmark queries are Flickr8k captions. Everything above about free-form queries is example-based. Image, text + image and More like this searches have no accuracy number at all.

## What I'd do next

1. **Add a "no strong match" note.** For image queries, use a cutoff around 0.70-0.75: every synthetic test image scored below 0.68, and 95% of real photos have a nearest neighbour at 0.753 or above. For text, show at most a soft hint below about 0.10, since that cutoff misfires on real cat and laptop photos.
2. **Warn when a query is truncated.** The token count is known before encoding, so the 64-token cutoff can be shown.
3. **Collapse duplicates in results.** The byte-identical pair is easy. Near-duplicates need a careful threshold, because different white birds over water already score about 0.968.
4. **Make result labels honest.** Mark the shown caption as "closest caption", or show the photo's first caption and put the closest one on hover.
5. **Score free-form queries.** Hand-label relevance for the 30 natural queries in `latency.json` and the probe queries, so accuracy is not measured only on Flickr8k-style paraphrases.
6. **Re-run the ViT-B-32 and MobileCLIP2-S2 benchmark rows** from fresh builds, and **time on the Space** what has only been timed locally: image queries, BLIP, and a wake from sleep.
7. **Trim memory.** Storing the SigLIP 2 weights in fp16 would halve their 1,431.2 MiB (arithmetic, not measured). 196.6M of the text tower's 282.3M parameters are the vocabulary table.
8. **Preload the models on the Space.** Listing the SigLIP 2 and BLIP repos under `preload_from_hub` in the README front matter would download them at build time, shortening the cold start and the first image upload.
9. **For a much larger collection,** pull candidates with an ANN index (FAISS IVF or HNSW) and keep the same z-score blending on top. At 8,091 photos, exact ranking takes 2.1-2.4 ms median locally (p95 up to 4.8 ms) and needs no tuning.

## Reproduce

From the repository root (Python 3.11):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt     # app requirements plus umap-learn, tqdm, faiss-cpu, requests

# Run the app. Uses data/ if present, otherwise downloads *.json + *.npy from the HF dataset.
streamlit run app/frontend/Home.py

# Rebuild the index from Flickr8k (writes data/: embeddings, thumbnails, UMAP projection)
python scripts/build_index.py --images /path/to/Flickr8k/Images

# Accuracy: 20,218 held-out caption queries over 8,091 photos (49.5 s locally)
python scripts/eval_retrieval.py data

# Compare another open_clip model
python scripts/build_index.py --images /path/to/Flickr8k/Images \
    --model ViT-B-32 --pretrained laion2b_s34b_b79k --out /tmp/b32 --no-thumbs
python scripts/eval_retrieval.py /tmp/b32

# Latency: mps (Apple GPU), cpu2 (2 CPU threads) and the old-app replay -> portfolio-screenshots/latency.json
python scripts/benchmark_latency.py
python scripts/benchmark_latency.py --configs cpu2 --skip-blip    # re-runs one config; the others in latency.json are kept
```

Notes:

- The `mps` configuration needs Apple silicon.
- The `old` configuration downloads the old index from Space commit `44167ce` and fetches images from Google Drive, so its timings depend on your connection.

Where the numbers in this document come from:

- **Local M5 latency:** `portfolio-screenshots/latency.json`.
- **Live Space timings:** `portfolio-screenshots/live_space.json`.
- **Accuracy:** `scripts/eval_retrieval.py`.
- **Model, data, memory and limit probes:** one-off scripts run against `SearchEngine` on `data/`. Those scripts are not in the repo.
