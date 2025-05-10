import numpy as np
import faiss
import json
import logging
import openai
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path

# Configuration
INDEX_PATH = "embedded_chunks/index.faiss"
METADATA_PATH = "embedded_chunks/metadata.json"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
TOP_K = 5
MAX_TOKENS = 512

logging.basicConfig(
  level=logging.INFO,
  format="[%(asctime)s] %(levelname)s: %(message)s",
  datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Setup
log.info("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

log.info("Loading metadata...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
  metadata = json.load(f)

log.info("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

log.info("Loading generation model...")
gen_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL, device_map="auto")
generator = pipeline("text-generation", model=gen_model, tokenizer=gen_tokenizer)

def search_faiss(query: str, top_k: int = 5) -> List[dict]:
  """Embed the query and return top-K metadata chunks from FAISS."""
  embedded = embedder.encode([f"Represent this passage for retrieval: {query}"])
  embedded = np.array(embedded).astype("float32")
  D, I = index.search(embedded, top_k)
  return [metadata[i] for i in I[0]]

def generate_via_openai(context: str, question: str) -> str:
  resp = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
      {"role": "system", "content": "You're a expert assistant answering homelab questions based on provided context."},
      {"role": "user", "content": f"Answer the question using the following context:\n\n{context}\n\nQuestion: {question}"}
    ],
    max_tokens=512,
    temperature=0.7,
  )
  return resp.choices[0].message.content.strip()

# FastAPI
app = FastAPI(
  title="Kelcode Homelab RAG Service",
  description="Retrieval-augmented demo for homelab docs, backed by OpenAI",
  version="0.1.0",
  docs_url="/docs",
  redoc_url="/redoc.html",
  openapi_url="/openapi.json",
)

class QuestionRequest(BaseModel):
  question: str
  top_k: int = TOP_K

@app.post("/qa")
async def answer_question(req: QuestionRequest):
  log.info(f"/qa request: '{req.question}'")
  query = req.question
  top_k = req.top_k
  retrieved = search_faiss(query, top_k)
  log.info(f"Retrieved {len(retrieved)} chunks from FAISS")

  # Build prompt from chunks
  context_blocks = []
  for entry in retrieved:
    context_blocks.append(f"### {entry['doc_title']}\n{entry['text']}\n")

  context = "\n".join(context_blocks)

  answer = generate_via_openai(context, req.question)

  return {
    "answer": answer,
    "sources": [{"title": r["doc_title"], "source": r["source"]} for r in retrieved]
  }

@app.post("/chunks")
async def get_chunks(req: QuestionRequest):
  log.info(f"/chunks request: '{req.question}'")
  query = req.question
  top_k = req.top_k
  retrieved = search_faiss(query, top_k)
  log.info(f"Retrieved {len(retrieved)} chunks from FAISS")

  return {
    "question": query,
    "top_k": top_k,
    "chunks": [
      {
        "rank": i + 1,
        "title": doc["doc_title"],
        "source": doc["source"],
        "category": doc["category"],
        "chunk_index": doc["chunk_index"],
        "text": doc["text"]
      }
      for i, doc in enumerate(retrieved)
    ]
  }

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def home(request: Request):
  # renders templates/index.html
  return templates.TemplateResponse("index.html", {"request": request})
