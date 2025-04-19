# kelcode-ai-labs

A hands-on AI engineering lab series exploring the full AI lifecycle — from data and fine-tuning to model optimisation, debugging, and deployment. Designed for platform builders and curious engineers, this project blends real-world tools with practical examples across multiple tracks.

## 🧠 What's Inside

This repository supports an end-to-end AI curriculum, structured into four modules:

### 📦 Module 1: Foundations of AI Modelling
- AI lifecycle walkthrough
- Key terminology and system thinking
- Accuracy vs speed vs cost trade-offs

### 📊 Module 2: Data Science Track (Modelling Focus)
- Data cleaning and exploration
- ML & deep learning models
- LoRA fine-tuning and quantisation (FP16, INT8)
- Evaluation and debugging (ONNX, TensorRT)

### ⚙️ Module 3: AI Engineering Track (Platform Focus)
- Model packaging (ONNX, TorchScript)
- FastAPI + Triton inference APIs
- Kubernetes deployment via Helm
- Versioning and monitoring

### 🏗️ Module 4: AI Platform Architecture
- RAG systems and vector DBs
- Multi-tenant architectures
- Security, cost engineering, and MLOps patterns

## 📁 Directory Structure

```bash
kelcode-ai-labs/
├── curriculum/           # Curriculum structure and lab objectives
├── module-01/            # Foundational concepts and lifecycle labs
├── module-02/            # Modelling-focused labs (LoRA, quant, ONNX, etc.)
│   ├── lab04-optimum.py
│   └── lab05-eval.py
├── module-03/            # Engineering labs (APIs, inference, k8s)
├── module-04/            # Architecture and ops
├── models/               # Lightweight models, adapters, config files
├── datasets/             # Sample and synthetic datasets (CSV)
├── outputs/              # Evaluation artifacts (charts, matrices)
└── README.md
