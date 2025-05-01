# Lab 06 – Serving Quantised Models with FastAPI

A compact FastAPI service that can switch between three quantised Transformer backends dynamically at request time:

1. **ONNX INT8** via ONNX Runtime
2. **PyTorch dynamic-quantised INT8**
3. **TensorRT FP16** on GPU

It exposes:

- `GET  /health`
- `POST /predict` → top-1 category
- `POST /predict/top5` → top-5 categories
- Swagger UI at `/docs` and a minimal HTML/JS user interface at `/`

## 🚀 Prerequisites

- Python 3.11+
- (Optional) CUDA 11+ & NVIDIA drivers for TensorRT
- [Docker](https://docs.docker.com/get-docker/) if you plan to build the container

From the repo root:

```bash
cd data-science-track/06-serving-quantised-models
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 📂 Directory Layout

```
06-serving-quantised-models/
├── api.py             # FastAPI routes
├── main.py            # ASGI app, startup & dependency logic
├── settings.py        # Pydantic BaseSettings for config/.env
├── schemas.py         # Pydantic request/response models
├── drivers/           # Model‐specific driver implementations
│   ├── base.py
│   ├── onnx_driver.py
│   ├── torch_driver.py
│   └── trt_driver.py
├── models/            # Place your quantised model dirs here:
│   ├── onnx/
│   ├── torch/
│   └── trt_fp16/
├── requirements.txt
├── Dockerfile         # Multi-stage build for CPU & GPU
├── static/            # JS & CSS for the demo UI
└── templates/         # Jinja2 template for the `/` endpoint
```

## 🔧 Configuration

Copy or create a `.env` alongside `settings.py` to override defaults:

```ini
# .env example
CORS_URLS=http://localhost:8080,https://your.domain
DEFAULT_MODEL=onnx
```

Key settings in `settings.py`:

- `cors_urls` / `CORS_URLS`
- `default_model` / `DEFAULT_MODEL`
- Model paths & filenames under `models/`

## ▶️ Running Locally

```bash
# from data-science-track/06-serving-quantised-models
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

- Health check:  `curl http://localhost:8080/health`
- Predict (top-1):
  ```bash
  curl -X POST http://localhost:8080/predict \
       -H "Content-Type: application/json" \
       -d '{"text":"AI tutor rolls out in schools"}'
  ```
- Top-5:
  ```bash
  curl -X POST http://localhost:8080/predict/top5 \
       -H "Content-Type: application/json" \
       -d '{"text":"Climate report warns UK cities"}'
  ```
- Swagger UI:  [http://localhost:8080/docs](http://localhost:8080/docs)
- Demo UI:     [http://localhost:8080/](http://localhost:8080/)

## 🐳 Docker

```bash
# Build (CPU or GPU-capable image)
docker build -t kelcode/lab06-quant-server .

# Run CPU-only
docker run --rm -p 8080:8080 kelcode/lab06-quant-server

# Run with GPU
docker run --rm --gpus all -p 8080:8080 kelcode/lab06-quant-server
```

The container auto-detects CUDA and will load the TensorRT driver only if a GPU is present.

## 🔜 Next Steps

- Prepare your quantised models under `models/` (see article "Model Preparation" section)
- Add more backends (e.g. Triton, Ollama) by extending `drivers/base.py`
- Instrument Prometheus metrics, rate-limit your endpoints, and deploy to Kubernetes
- In Lab 07 we’ll build a Retrieval-Augmented QA demo on the same framework
