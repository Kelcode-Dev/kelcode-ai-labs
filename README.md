# Kelcode AI Labs

A hands-on AI engineering lab series exploring the full AI lifecycle — from data and fine-tuning to model optimisation, debugging, and deployment. Designed for platform builders and curious engineers, this project blends real-world tools with practical examples across multiple tracks.

Articles supporting this repo can be found at [kelcode.co.uk](https://kelcode.co.uk)

## 🧠 What's Inside

This repository supports an end-to-end AI curriculum, structured into four modules:

| Module | Focus                         | Status          |
|--------|-------------------------------|-----------------|
| 1      | Foundations of AI Modelling   | 🔜 Coming Soon  |
| 2      | Data Science & Modelling      | ✅ In progress  |
| 3      | AI Engineering & Deployment   | 🔜 Coming Soon  |
| 4      | Advanced AI Platform Arch.    | 🔜 Coming Soon  |

For the full details, check out the [CURRICULUM.md] doc

### 📦 Module 1: Foundations of AI Modelling
- AI lifecycle walkthrough
- Key terminology and system thinking
- Accuracy vs speed vs cost trade-offs

### 📊 Module 2: Data Science Track (Modelling Focus) [in-progress]
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

Each module with it's own set of labs and readmes supporting those particular labs

## 📁 Directory Structure

```bash
kelcode-ai-labs/
├── foundations-ai-modelling/     # Foundational concepts and lifecycle labs
├── data-science-track/           # Modelling-focused labs (LoRA, quant, ONNX, etc.)
│   ├── lab01
│   ├── lab02
│   ├── lab03
│   └── ...
├── ai-engineering-track/         # Engineering labs (APIs, inference, k8s)
├── ai-platform-architectures/    # Architecture and ops
├── CURRICULUM.md                 # Curriculum details, likely to change over time
└── README.md
```

## 📚 Articles in the Series

1. [Embarking on an AI Engineering Journey: From Data to Deployment](https://kelcode.co.uk/embarking-on-an-ai-engineering-journey)
2. [Machine Learning: Building Our First Classifier with Scikit-Learn](https://kelcode.co.uk/classical-machine-learning-iris-dataset-and-scikitlearn/)
3. [Deep Learning: Training Our First Model](https://kelcode.co.uk/deep-learning-training-our-first-model/)
4. [Fine-Tuning a Model with LoRA](https://kelcode.co.uk/fine-tuning-a-model-with-lora/)
5. [Quantisation in Deep Learning: A Practical Lab Guide](https://kelcode.co.uk/quantisation-in-deep-learning/)
6. [Debugging and Repairing Tensorrt Inference](https://kelcode.co.uk/debugging-and-repairing-tensorrt-inference/)
7. [Serving Quantised Models with FastAPI](https://kelcode.co.uk/serving-quantised-models-with-fastapi)
8. Real world Retrieval‐Augmented Q&A example - Coming Soon!

## Contributing

We welcome your feedback and improvements!
Please read our [Contributing Guide](CONTRIBUTING.md) for details on how to report issues, submit pull requests, and follow our commit message conventions.
