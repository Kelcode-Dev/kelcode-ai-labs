# 🧪 **Lab 07: Serving Quantised AI Models with FastAPI**

### 🎯 Purpose:
Build a custom FastAPI server that can serve predictions from your three quantised models (Torch INT8, ONNX INT8, TensorRT FP16). Focus on exposing a unified `/predict` endpoint with switchable backends.

### ✅ Core Prompts / Structure:

## Overview
Now that we’ve validated our quantised models, we need a clean way to expose them for use in downstream systems. This lab builds a FastAPI app to serve the models and return top-5 predictions.

## Goals
- Serve quantised models via an HTTP API
- Support backend switching between Torch, ONNX, and TensorRT
- Return predictions + confidence scores
- Include timing/latency per request

## Files
- `api/app.py` – FastAPI app
- `predict.py` – backend logic (imported or refactored from Lab 06)
- `config.py` – model paths, label map, backend toggle
- `requirements.txt` – FastAPI, Uvicorn, runtime deps

## API Endpoint
- `POST /predict`
- Input: `{"text": "headline string", "backend": "onnx"}` (optional backend param)
- Output: `{"top_5": [...], "inference_time": 0.053}`

## Optional Add-Ons
- Add CLI to start with selected backend only
- Return raw logits if debug flag is set
- Auto-reload on changes (for dev)

## What You’ll Learn
- How to serve an AI model as a microservice
- Backend abstraction and switching
- Latency profiling for real-world usage
