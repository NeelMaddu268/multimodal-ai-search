"""2-D UMAP map of the embedding space, with your query and its top matches overlaid."""

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Embedding Space", page_icon="🗺️", layout="wide")
st.title("🗺️ Embedding space")
with st.spinner("Loading the search model and index…"):
    from resources import embed_text, get_engine

    engine = get_engine()

st.caption(
    "Every photo and one caption per photo, projected from the model's embedding space to 2-D with UMAP. "
    "Photos and captions form separate continents: the *modality gap*. Search still works because the "
    "layout inside each continent lines up, so similar ideas sit in the same place on both."
)

if engine.projection is None:
    st.warning("No projection found. Rebuild the index without `--no-projection`.")
    st.stop()


@st.cache_data
def base_traces():
    n = len(engine)
    xy = engine.projection
    caption_text = [engine.captions[c] for c in engine.projected_captions]
    image_text = [engine.captions[engine.starts[i]] for i in range(n)]
    style = dict(mode="markers", hoverinfo="text")
    return [
        go.Scattergl(x=xy[:n, 0], y=xy[:n, 1], text=image_text, name="photos", marker=dict(size=3, color="#e4572e", opacity=0.45), **style),
        go.Scattergl(x=xy[n:, 0], y=xy[n:, 1], text=caption_text, name="captions", marker=dict(size=3, color="#4c78a8", opacity=0.45), **style),
    ]


query = st.text_input("Drop a query onto the map", placeholder="e.g. a dog swimming in a lake")
fig = go.Figure(base_traces())

if query.strip():
    vec = embed_text(query.strip())
    hits = engine.search(text_vec=vec, k=10)
    hit_xy = engine.projection[[h.image for h in hits]]
    q_xy = engine.project(vec)
    fig.add_trace(go.Scattergl(
        x=hit_xy[:, 0], y=hit_xy[:, 1], text=[h.caption for h in hits], name="top 10 photos", mode="markers",
        hoverinfo="text", marker=dict(size=12, color="#ffb000", line=dict(width=1.5, color="black")),
    ))
    fig.add_trace(go.Scattergl(
        x=[q_xy[0]], y=[q_xy[1]], text=[f"Query: {query}"], name="your query", mode="markers", hoverinfo="text",
        marker=dict(size=18, symbol="star", color="#2ca02c", line=dict(width=1.5, color="black")),
    ))
    st.caption("The star is placed among the captions nearest to your query; the gold dots are the photos the search returned.")

fig.update_layout(
    height=720, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.02, x=0, itemsizing="constant"),
    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
)
st.plotly_chart(fig, width="stretch")
