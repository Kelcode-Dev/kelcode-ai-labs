# 🧪 Lab 04 – Quantised Model Inference Benchmarking

In this lab we explore INT8 and FP16 quantisation of a LoRA-fine-tuned BERT classifier (9 labels: AG News + custom categories) across three runtimes. By the end you’ll have concrete numbers for:

- **Model size** (MB on disk)
- **Inference latency** (ms per sample)
- **Throughput** (samples/sec)
- **Accuracy** (nine-way F1 and prediction consistency)

## 🧠 What Is Quantisation?

Quantisation trades a little accuracy for drastic gains in size and speed by converting FP32 weights into:

- **INT8** → 8-bit integers (smallest footprint and fastest on CPU)
- **FP16** → 16-bit floats (half-precision, GPU-accelerated via Tensor Cores)

These lower-precision formats unlock leaner matrix multiplies and lower memory bandwidth.

## 🔁 Workflow Overview

We benchmark three pipelines against the **same LoRA-trained weights**:

1. **PyTorch Dynamic INT8 (CPU)**
   One line of code; no calibration required
2. **ONNX Runtime Dynamic INT8 (CPU & Cloud)**
   Framework-agnostic INT8 for C#, C++, containerised services
3. **TensorRT Mixed-Precision FP16 (GPU)**
   Half-precision Tensor Core engine for blistering throughput

Each script produces a model directory under `quantised-model-*/`, and `predict.py` runs all three for side-by-side comparison.

## 📈 Benchmarking Methodology

`predict.py`:

- Initialises a PyCUDA context for GPU runs
- Loads each backend: PyTorch via `torch.load`, ONNX via `InferenceSession`, TensorRT via `trt.Runtime`
- Defines three `predict_*` functions that tokenise inputs, run inference and time each call
- Applies a softmax to logits, prints top-5 predictions and times; then generates a bar chart per prompt

All three backends are exercised on identical input texts so our results reflect only differences in precision and runtime.

## ⚡ Inference Comparison (Example)

```text
Predicting: 'Cybersecurity breach exposes NHS patient records'

[PyTorch Quantised]: security (7) | 0.4846 sec
Top 5 Predictions:
  - security     0.7537
  - health       0.2129
  - education    0.0143
  - tech-policy  0.0100
  - climate      0.0090

[ONNX Quantised]: security (7) | 0.0133 sec
Top 5 Predictions:
  - security     0.5561
  - health       0.3933
  - education    0.0300
  - climate      0.0118
  - tech-policy  0.0088

[TensorRT FP16]: Business (0) | 0.0085 sec
Top 5 Predictions:
  - Business     0.9292
  - Sports       0.0637
  - Sci/Tech     0.0067
  - World        0.0003
  - security     0.0000

>> Saved prediction confidence comparison to: confidence_cybersecurity_breach.png
```

## 📊 Results

| Backend                  | Precision | Device | Size (MB) | Latency (ms) | Throughput (samples/sec) | Accuracy on 3 test headlines |
|--------------------------|-----------|--------|----------:|-------------:|--------------------------:|------------------------------|
| PyTorch Dynamic INT8     | INT8      | CPU    |       174 |          ~400 |                      ~2.5 | 3/3 correct                  |
| ONNX Runtime Dynamic INT8| INT8      | CPU    |       106 |           ~13 |                     ~73.6 | 3/3 correct                  |
| TensorRT Mixed-Precision FP16 | FP16      | GPU    |       211 |            ~4 |                    ~242   | 0/3 correct*                 |

_NB: TensorRT FP16 misclassified all three headlines due to a label-mapping/input-binding mismatch in this iteration. We’ll fix it in Lab 06 and re-benchmark for true accuracy comparison_

### Confidence Visualisations

For each headline we plot the top-5 softmax scores:

- `confidence_cybersecurity_breach.png`
- `confidence_ai_tutor_program_set.png`
- `confidence_climate_report_warns.png`

These charts reveal how quantisation and runtime can shift confidence distributions.

## 🧠 Key Takeaways

- **PyTorch INT8** is your zero-calibration, low-memory CPU champion
- **ONNX Runtime INT8** adds portability for C#/C++ and cloud services with identical speed/accuracy
- **TensorRT FP16** on GPU delivers the highest throughput once label mapping is fixed

Quantisation can shift logits subtly, so always validate predictions and inspect confidence plots before production.

## 🛠 Troubleshooting Notes

- **TensorRT mislabels** → verify engine bindings and reload the original `config.json` for correct `id2label`
- **CUDA context errors** → wrap GPU code in `try/finally` and always call `cuda_ctx.pop()` and `cuda_ctx.detach()`
- **Shape or dtype mismatches** → confirm dynamic axes (`--minShapes`, `--optShapes`, `--maxShapes`) match your ONNX export

## 📁 Project Structure

```
.
├── lab04-torch.py
├── lab04-optimum.py
├── lab04-nvidia.py
├── predict.py
├── quantised-model-torch/
├── quantised-model-optimum/
├── quantised-model-tensorrt/
├── confidence_*.png
├── requirements.txt
└── README.md
```
