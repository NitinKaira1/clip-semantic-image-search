import streamlit as st
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

st.set_page_config(page_title="Semantic Image Search (CLIP)", page_icon="🔍", layout="wide")

import os
_LOCAL_MODEL_PATH = "clip-local"
MODEL_NAME = _LOCAL_MODEL_PATH if os.path.isdir(_LOCAL_MODEL_PATH) else "openai/clip-vit-base-patch32"


@st.cache_resource(show_spinner="Loading CLIP model (first run only)...")
def load_model():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


model, processor = load_model()


def _extract_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


def get_image_embeddings(images):
    inputs = processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeds = _extract_tensor(model.get_image_features(**inputs))
    embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
    return embeds


def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        embeds = _extract_tensor(model.get_text_features(**inputs))
    embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
    return embeds


st.title("🔍 Semantic Image Search Engine")
st.markdown(
    "Search a gallery of images using **plain English** — powered by OpenAI's "
    "**CLIP** model. No labels, no training, no keywords required."
)

with st.sidebar:
    st.header("1. Build your image gallery")
    uploaded_files = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    st.caption("Tip: upload 10–30 varied images (people, objects, scenes) for a good demo.")
    st.markdown("---")
    st.markdown(
        "**How it works:** every image and your text query are converted into "
        "512-dimensional vectors by CLIP. Images whose vector is closest "
        "(cosine similarity) to your query's vector are the best matches."
    )

if uploaded_files:
    current_names = [f.name for f in uploaded_files]
    if st.session_state.get("gallery_names") != current_names:
        images = [Image.open(f).convert("RGB") for f in uploaded_files]
        with st.spinner("Indexing images..."):
            embeddings = get_image_embeddings(images)
        st.session_state["gallery"] = images
        st.session_state["embeddings"] = embeddings
        st.session_state["gallery_names"] = current_names

    st.header("2. Search")
    query = st.text_input(
        "Describe what you're looking for", placeholder="e.g. a red car on a street"
    )
    max_k = min(10, len(uploaded_files))
    top_k = st.slider("Number of results", 1, max_k, min(5, max_k))

    if query:
        text_embed = get_text_embedding(query)
        sims = (st.session_state["embeddings"] @ text_embed.T).squeeze(1).numpy()
        top_idx = np.argsort(-sims)[:top_k]

        cols = st.columns(min(top_k, 5))
        for rank, idx in enumerate(top_idx):
            with cols[rank % len(cols)]:
                st.image(st.session_state["gallery"][idx], use_container_width=True)
                st.caption(f"Score: {sims[idx]:.3f}")
else:
    st.info("👈 Upload some images in the sidebar to get started.")
    st.markdown(
        "No dataset handy? Try grabbing a free sample set from "
        "[Unsplash](https://unsplash.com/) or [Pexels](https://www.pexels.com/)."
    )