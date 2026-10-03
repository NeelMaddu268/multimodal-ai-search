"""Cached models and encoders shared by the Streamlit pages."""

import io

import streamlit as st
import torch
from PIL import Image

from engine import SearchEngine


@st.cache_resource(show_spinner=False)  # the pages show their own spinner
def get_engine():
    return SearchEngine()


@st.cache_resource(show_spinner="Loading the BLIP captioner…")
def get_captioner():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    name = "Salesforce/blip-image-captioning-base"
    return BlipProcessor.from_pretrained(name), BlipForConditionalGeneration.from_pretrained(name).eval()


@st.cache_data(max_entries=256, show_spinner=False)
def embed_text(text):
    return get_engine().encode_text(text)


@st.cache_data(max_entries=32, show_spinner=False)
def embed_upload(data):
    return get_engine().encode_image(Image.open(io.BytesIO(data)))


@st.cache_data(max_entries=32, show_spinner=False)
def describe(data):
    """BLIP caption for an uploaded image."""
    processor, model = get_captioner()
    with torch.no_grad():
        out = model.generate(**processor(Image.open(io.BytesIO(data)).convert("RGB"), return_tensors="pt"), max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)
