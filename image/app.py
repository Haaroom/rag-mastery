import faiss
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
# 1. Initialize CLIP Model and Processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# 2. Setup your image collection 
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
# 3. Instantiate FAISS Index for Inner Product (Cosine similarity when normalized)
index = faiss.IndexFlatIP(512)
embeddings = []
# 4. Generate embeddings for the database
if image_paths:
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            # Normalize vector to unit length
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy())
    # Stack lists into a single 2D float32 numpy array
    embeddings_array = np.vstack(embeddings).astype("float32")
    index.add(embeddings_array)
else:
    print("Warning: Please populate 'image_paths' with your data.")
# 5. Query the Index
query_image = Image.open("query.jpg").convert("RGB")
inputs = processor(images=query_image, return_tensors="pt")
with torch.no_grad():
    query_emb = model.get_image_features(**inputs)
    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
# Ensure query is float32 numpy array for FAISS matching
query_np = query_emb.cpu().numpy().astype("float32")
# Search the top K matches
if index.ntotal > 0:
    scores, indices = index.search(query_np, k=min(5, index.ntotal))
    print("Scores:", scores)
    print("Indices:", indices)
