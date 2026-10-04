"""Multimodal AI Search: find Flickr8k photos by describing them, by example image, or both."""

import time

import numpy as np
import streamlit as st
from PIL import UnidentifiedImageError

st.set_page_config(page_title="Multimodal AI Search", page_icon="🔍", layout="wide")
st.html("""<style>
  [data-testid="stImage"] img { aspect-ratio: 4 / 3; object-fit: cover; border-radius: 6px; }
  .score { color: #888; font-size: 0.75rem; }
  [data-testid="stButtonGroup"] > div { flex-wrap: wrap; }
</style>""")

EXAMPLES = [
    "dog catching a frisbee",
    "kids in a fountain",
    "surfer on a big wave",
    "rock climbing",
    "concert at night",
    "snowy mountain hike",
]
COLUMNS = 4

st.title("🔍 Multimodal AI Search")
with st.spinner("Loading the search model and index…"):
    # imported here so the page shows something while torch loads on a cold start
    from resources import describe, embed_text, embed_upload, get_engine, open_upload

    engine = get_engine()
st.caption(f"Search {len(engine):,} photos by describing them, by showing an example, or both.")

state = st.session_state
state.setdefault("saved_query", "")
if "query" not in state:  # Streamlit drops widget state while another page runs; bring the search back
    state.query = state.saved_query
state.setdefault("similar_to", None)
state.setdefault("history", [])
state.setdefault("seed", 0)


def remember(query):
    query = query.strip()
    if query:
        state.history = [query] + [h for h in state.history if h != query][:9]


def on_query_change():
    state.saved_query = state.query
    state.similar_to = None
    state.example = None  # pills act as buttons
    remember(state.query)


def set_query(query):
    state.query = query
    on_query_change()


def show_similar(i):
    state.similar_to = i


def clear_similar():
    state.similar_to = None


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    k = st.select_slider("Results", options=[8, 12, 16, 24, 32, 48], value=12)
    mode = st.segmented_control(
        "Text ranking",
        ["hybrid", "visual", "captions"],
        default="hybrid",
        required=True,
        format_func=str.capitalize,
        help="Visual compares your words directly to the pixels. Captions compares them to the human-written "
        "descriptions of each photo (usually five). Hybrid blends both and scores best on the benchmark in the README.",
    )
    if state.history:
        st.subheader("Recent searches")
        for h in state.history:
            st.button(h, key=f"hist-{h}", on_click=set_query, args=(h,), type="tertiary")
    st.divider()
    st.caption(f"**Model:** {engine.model_name}  \n**Index:** {len(engine):,} photos, {len(engine.captions):,} captions (Flickr8k)")
    st.caption("[Source on GitHub](https://github.com/NeelMaddu268/multimodal-ai-search)")


# ---------------- query inputs ----------------
left, right = st.columns([3, 2], gap="large")
with left:
    st.text_input(
        "Describe a photo",
        key="query",
        placeholder="e.g. a dog catching a frisbee at the beach",
        on_change=on_query_change,
    )
    st.pills("Try", EXAMPLES, key="example", on_change=lambda: set_query(state.example or state.query), label_visibility="collapsed")
with right:
    upload = st.file_uploader("…and/or start from an image", type=["jpg", "jpeg", "png", "webp"], on_change=clear_similar)

query = state.query.strip()
upload_bytes = upload.getvalue() if upload else None
if upload_bytes:
    try:
        upload_image = open_upload(upload_bytes)
    except (UnidentifiedImageError, OSError):
        right.error("Couldn't read that file as an image.")
        upload_bytes = None

if upload_bytes:
    with right:
        preview, about = st.columns([1, 2])
        preview.image(upload_image, width="stretch")
        with about, st.spinner("Captioning…"):
            st.markdown(f"**BLIP caption:** {describe(upload_bytes)}")

image_share = 0.5
if query and upload_bytes:
    image_share = st.slider("Text ↔ image balance", 0.0, 1.0, 0.5, 0.05, help="0 = only your text matters, 1 = only your image matters")


# ---------------- search ----------------
def render(results):
    for row in range(0, len(results), COLUMNS):
        cols = st.columns(COLUMNS)
        for j, (col, r) in enumerate(zip(cols, results[row:row + COLUMNS])):
            with col, st.container(border=True):
                st.image(engine.thumbnail(r.image), width="stretch")
                st.markdown(f"<small>{r.caption}</small>", unsafe_allow_html=True)
                signals = (("visual", r.visual), ("caption", r.caption_score), ("image", r.image_score))
                scores = [f"{name} {v:.3f}" for name, v in signals if not np.isnan(v)]
                if scores:
                    st.markdown(f'<span class="score">#{row + j + 1} · {" · ".join(scores)}</span>', unsafe_allow_html=True)
                st.button("More like this", key=f"sim-{r.image}", on_click=show_similar, args=(r.image,), width="stretch")


start = time.perf_counter()
if state.similar_to is not None:
    i = state.similar_to
    results = engine.search(image_vec=engine.image_emb[i], k=k, exclude=i)
    head, back = st.columns([6, 1], vertical_alignment="center")
    with head:
        thumb, text = st.columns([1, 6], vertical_alignment="center")
        thumb.image(engine.thumbnail(i), width="stretch")
        text.markdown(f"**Photos that look like this one**  \n<small>{engine.captions[engine.starts[i]]}</small>", unsafe_allow_html=True)
    back.button("✕ Clear", on_click=clear_similar, width="stretch")
elif query or upload_bytes:
    results = engine.search(
        text_vec=embed_text(query) if query else None,
        image_vec=embed_upload(upload_bytes) if upload_bytes else None,
        image_share=image_share,
        mode=mode,
        k=k,
    )
else:
    results = None

if results is None:
    head, shuffle = st.columns([6, 1], vertical_alignment="center")
    head.markdown("**Explore the collection** — or click *More like this* on any photo.")
    if shuffle.button("Shuffle", width="stretch"):
        state.seed += 1
    picks = np.random.default_rng(state.seed).choice(len(engine), size=k, replace=False)
    results = [engine.result(int(i)) for i in picks]
else:
    st.caption(f"{len(results)} results in {(time.perf_counter() - start) * 1000:.0f} ms")

render(results)
