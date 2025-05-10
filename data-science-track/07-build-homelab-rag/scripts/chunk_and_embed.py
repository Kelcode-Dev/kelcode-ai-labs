import json
import os
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer

# CONFIG
PARAPHRASED_FILE = Path("paraphrased_documents/mixtral_paraphrased.jsonl")
INDEX_DIR = Path("embedded_chunks")
INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 100
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# Ensure output directory exists
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Load model + tokenizer
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
  tokens = tokenizer.encode(text, add_special_tokens=False)
  chunks = []
  for i in range(0, len(tokens), chunk_size - overlap):
    chunk_tokens = tokens[i:i + chunk_size]
    chunk_text = tokenizer.decode(chunk_tokens)
    chunks.append(chunk_text)
    if i + chunk_size >= len(tokens):
      break
  return chunks

# Load paraphrased documents
print("Loading paraphrased documents...")
docs = []
with open(PARAPHRASED_FILE, "r", encoding="utf-8") as f:
  for line in f:
    doc = json.loads(line)
    docs.append(doc)

# Process + embed
print("Chunking and embedding...")
all_embeddings = []
metadata = []

for doc in tqdm(docs, desc="Embedding chunks"):
  chunks = chunk_text(doc["paraphrased_content"])
  prefixed_chunks = [f"Represent this passage for retrieval: {chunk}" for chunk in chunks]
  embeddings = embedder.encode(prefixed_chunks)

  for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    metadata.append({
      "doc_title": doc["title"],
      "source": doc["source"],
      "category": doc["category"],
      "chunk_index": i,
      "text": chunk
    })
    all_embeddings.append(embedding)

# Build FAISS index
print(f"Saving {len(all_embeddings)} embeddings to FAISS index...")
dimension = len(all_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(all_embeddings).astype("float32"))

faiss.write_index(index, str(INDEX_FILE))
with open(METADATA_FILE, "w", encoding="utf-8") as f:
  json.dump(metadata, f, indent=2)

print(f"\nIndex saved to {INDEX_FILE}")
print(f"Metadata saved to {METADATA_FILE}")
