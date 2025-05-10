# Homelab RAG Service

A self-hosted Retrieval-Augmented Generation (RAG) demo for your homelab docs. This repo shows how to:

1. **Paraphrase** source docs into a clean, uniform JSONL
2. **Chunk & embed** them with `BAAI/bge-base-en-v1.5`
3. **Index** embeddings in FAISS for ultra-fast similarity search
4. **Serve** a QA API + simple web UI with FastAPI + OpenAI/Mistral

## 🚀 Features

- **Synthetic paraphrasing** pipeline to normalize your documents
- **Configurable chunk size & overlap** for maximal retrieval recall
- **Batch embedding** via Sentence-Transformers + custom “represent” prompt
- **FAISS** vector store (flat L2) for instant k-NN lookup
- **FastAPI** endpoints:
  - `POST /qa` – answer questions with contextual retrieval + LLM
  - `POST /chunks` – inspect raw chunks returned by FAISS
- **Minimal web UI** (HTML/JS/CSS) for interactive question answering

## 🛠️ Prerequisites

- Python 3.10+
- An **OpenAI API key** (if using OpenAI for generation)
- GPU recommended for local LLM inference

## 📥 Installation

1. **Clone** this repo
  ```bash
  git clone https://github.com/Kelcode-Dev/kelcode-ai-labs
  cd data-science-track/07-build-homelab-rag
  ```

2. **Create a venv & install deps**

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

3. **Set** your OpenAI key

  ```bash
  export OPENAI_API_KEY="sk-..."
  ```

## 🔄 Data Prep

Run the seed and paraphrase scripts to assemble your JSONL of homelab docs:

```bash
python seed_documents/seed_homelab_docs.py
python scripts/paraphrase_with_mixtral.py
```

## 🔗 Chunk & Embed

Split each paraphrased doc into overlapping passages, batch‐encode them with a retrieval prompt, then build a FAISS index:

```bash
python scripts/chunk_and_embed.py
```

This will write:

* `embedded_chunks/index.faiss`
* `embedded_chunks/metadata.json`

## 🚀 Serve & Query

Start the FastAPI app:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

Use `curl` or the minimal web UI at `http://localhost:8080`:

```bash
# QA
curl -X POST http://localhost:8080/qa \
  -H "Content-Type: application/json" \
  -d '{"question":"How to set up Grafana with Postgres?"}'

# Inspect raw chunks
curl -X POST http://localhost:8080/chunks \
  -H "Content-Type: application/json" \
  -d '{"question":"Docker logging with Loki"}'
```

## 📂 Directory Structure

```
.
├── app.py
├── embedded_chunks
│   ├── index.faiss
│   └── metadata.json
├── paraphrased_documents
│   └── mixtral_paraphrased.jsonl
├── seed_documents
│   ├── seed_docs.jsonl
│   └── seed_homelab_docs.py
├── scripts
│   ├── chunk_and_embed.py
│   └── paraphrase_with_mixtral.py
├── static
│   ├── rag.js
│   └── styles.css
├── templates
│   └── index.html
├── requirements.txt
└── README.md
```

## 🎯 Next Steps

In the **next article**, we’ll cover **testing & validating** your RAG pipeline:

* Evaluate retrieval quality (recall\@k, human review)
* Measure end-to-end performance
* Fine-tune prompts & models for your homelab docs

## 🤝 Contributing

Feel free to open issues or submit PRs. Let’s make homelab RAG rock!

**Key updates**
- Paths under `scripts/` and `seed_documents/` now match your tree
- Removed the Docker/Mongo mention since it isn’t used here
- Unified port (`8080`) across serve & example calls
- Added a brief **Contributing** section to invite feedback
