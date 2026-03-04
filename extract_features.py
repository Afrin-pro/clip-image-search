import os
import numpy as np
import torch
import clip
import faiss
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "256_ObjectCategories"
INDEX_PATH = "image_index.faiss"
PATHS_PATH = "image_paths.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# Load CLIP Model
# -----------------------------
print("Loading CLIP model (ViT-L/14)...")
model, preprocess = clip.load("ViT-L/14", device=device)

# -----------------------------
# Collect Image Paths
# -----------------------------
image_paths = []

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(".jpg"):
            image_paths.append(os.path.join(root, file))

print("Total Images Found:", len(image_paths))

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(paths):
    features = []

    for path in tqdm(paths):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model.encode_image(image)
                feature /= feature.norm(dim=-1, keepdim=True)

            features.append(feature.cpu().numpy())

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return np.vstack(features)


print("Extracting features...")
image_features = extract_features(image_paths)

print("Feature shape:", image_features.shape)

# -----------------------------
# Create FAISS Index
# -----------------------------
dimension = image_features.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity
index.add(image_features)

print("FAISS index created!")

# -----------------------------
# Save Files
# -----------------------------
faiss.write_index(index, INDEX_PATH)
np.save(PATHS_PATH, image_paths)

print("Saved:")
print("- image_index.faiss")
print("- image_paths.npy")
print("Done ✅")