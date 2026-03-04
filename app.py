import streamlit as st
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import os

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="CLIP Image Search", layout="wide")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# LOAD MODEL (Cached)
# ---------------------------
@st.cache_resource
def load_model():
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess

model, preprocess = load_model()

# ---------------------------
# LOAD FAISS INDEX
# ---------------------------
@st.cache_resource
def load_index():
    index = faiss.read_index("image_index.faiss")
    image_paths = np.load("image_paths.npy", allow_pickle=True)
    return index, image_paths

index, image_paths = load_index()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("⚙ Settings")
top_k = st.sidebar.slider("Top-K Results", 1, 20, 5)

search_mode = st.sidebar.radio(
    "Search Mode",
    ["Image Search", "Text Search"]
)

# ---------------------------
# IMAGE SEARCH
# ---------------------------
def search_by_image(image, top_k):
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model.encode_image(image)
        feature /= feature.norm(dim=-1, keepdim=True)

    feature = feature.cpu().numpy()
    similarities, indices = index.search(feature, top_k)

    return similarities[0], indices[0]

# ---------------------------
# TEXT SEARCH
# ---------------------------
def search_by_text(query, top_k):
    text = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_features = text_features.cpu().numpy()
    similarities, indices = index.search(text_features, top_k)

    return similarities[0], indices[0]

# ---------------------------
# UI
# ---------------------------
st.title("🔍 CLIP + FAISS Image Search Engine")

if search_mode == "Image Search":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Query Image", width=300)

        similarities, indices = search_by_image(image, top_k)

        st.subheader("Results")
        cols = st.columns(top_k)

        for i, idx in enumerate(indices):
            result_image = Image.open(image_paths[idx])
            cols[i].image(result_image, caption=f"{similarities[i]:.2f}")

else:
    query = st.text_input("Enter text (e.g., 'a deer in forest')")

    if query:
        similarities, indices = search_by_text(query, top_k)

        threshold = 0.25
        filtered = [(sim, idx) for sim, idx in zip(similarities, indices) if sim >= threshold]

        if len(filtered) == 0:
            st.warning("No results above threshold. Try lowering it.")
        else:
            st.subheader("Results")
            cols = st.columns(len(filtered))

            for i, (sim, idx) in enumerate(filtered):
                result_image = Image.open(image_paths[idx])
                cols[i].image(result_image, caption=f"{sim:.2f}")