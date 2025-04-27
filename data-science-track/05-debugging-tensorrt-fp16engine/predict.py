import os
import time
import torch
import onnxruntime as ort
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tabulate import tabulate
from pathlib import Path
from collections import defaultdict

# Config
TEXTS = [
  ("Cybersecurity breach exposes NHS patient records", "security"),
  ("AI tutor program set to roll out in Scottish schools", "education"),
  ("Climate report warns UK cities face extreme flooding by 2030", "climate")
]
LABELS = ['World', 'Sports', 'Business', 'Sci/Tech', 'climate', 'education', 'health', 'security', 'tech-policy']
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
cuda.Device(0).make_context()
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# Paths
BASE_DIR = Path("../../04-model-optimisation").resolve()
LORA_MODEL_PATH = Path(os.path.join(BASE_DIR, "lora/lora-news/full-model")).resolve()
OPTIMUM_ONNX_PATH = Path(os.path.join(BASE_DIR, "quantisation/quantised-model-optimum/model_quantised.onnx")).resolve()
QUANTISED_TORCH_MODEL_PATH = Path(os.path.join(BASE_DIR, "quantisation/quantised-model-torch/quantised_model.pt")).resolve()
TENSORRT_ONNX_PATH = Path("requantised-tensorrt/model.onnx").resolve()
TENSORRT_ENGINE_PATH = Path("requantised-tensorrt/model_fp16.engine").resolve()

# Load Tokenizer and Models
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_PATH, local_files_only=True)

lora_model = AutoModelForSequenceClassification.from_pretrained(LORA_MODEL_PATH, local_files_only=True, use_safetensors=True)
lora_model.eval()

quantised_torch_model = torch.load(QUANTISED_TORCH_MODEL_PATH, map_location="cpu", weights_only=False)
quantised_torch_model.eval()

onnx_session_optimum = ort.InferenceSession(str(OPTIMUM_ONNX_PATH), providers=["CPUExecutionProvider"])
onnx_session_tensorrt = ort.InferenceSession(str(TENSORRT_ONNX_PATH), providers=["CPUExecutionProvider"])

with open(TENSORRT_ENGINE_PATH, "rb") as f:
  trt_engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

# Softmax
def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

# Inference Functions
def predict_lora(text):
  inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
  with torch.no_grad():
    logits = lora_model(**inputs).logits.squeeze().numpy()
  return logits

def predict_torch(text):
  inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
  with torch.no_grad():
    logits = quantised_torch_model(**inputs).logits.squeeze().numpy()
  return logits

def predict_onnx_optimum(text):
  inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
  input_ids = inputs["input_ids"].astype(np.int64)
  attention_mask = inputs["attention_mask"].astype(np.int64)
  token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
  ort_inputs = {
    onnx_session_optimum.get_inputs()[0].name: input_ids,
    onnx_session_optimum.get_inputs()[1].name: attention_mask,
    onnx_session_optimum.get_inputs()[2].name: token_type_ids
  }
  logits = onnx_session_optimum.run(None, ort_inputs)[0].squeeze()
  return logits

def predict_onnx_tensorrt(text):
  inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
  input_ids = inputs["input_ids"].astype(np.int32)
  attention_mask = inputs["attention_mask"].astype(np.int32)
  token_type_ids = np.zeros_like(input_ids, dtype=np.int32)
  ort_inputs = {
    onnx_session_tensorrt.get_inputs()[0].name: input_ids,
    onnx_session_tensorrt.get_inputs()[1].name: attention_mask,
    onnx_session_tensorrt.get_inputs()[2].name: token_type_ids
  }
  logits = onnx_session_tensorrt.run(None, ort_inputs)[0].squeeze()
  return logits

def predict_trt(text):
  context = trt_engine.create_execution_context()
  inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
  input_ids = inputs["input_ids"].astype(np.int32)
  attention_mask = inputs["attention_mask"].astype(np.int32)
  token_type_ids = np.zeros_like(input_ids, dtype=np.int32)

  context.set_input_shape("input_ids", input_ids.shape)
  context.set_input_shape("attention_mask", attention_mask.shape)
  context.set_input_shape("token_type_ids", token_type_ids.shape)

  bindings = []
  device_buffers = []
  for inp in [input_ids, attention_mask, token_type_ids]:
    d_in = cuda.mem_alloc(inp.nbytes)
    cuda.memcpy_htod(d_in, inp)
    bindings.append(int(d_in))
    device_buffers.append(d_in)

  output_shape = context.get_tensor_shape("logits")
  output = np.empty(output_shape, dtype=np.float32)
  d_out = cuda.mem_alloc(output.nbytes)
  bindings.append(int(d_out))
  device_buffers.append(d_out)

  context.execute_v2(bindings)
  cuda.memcpy_dtoh(output, d_out)
  return output.squeeze()

def plot_latency(results):
  latency_data = defaultdict(list)
  for res in results:
      latency_data[res["model"]].append(res["prediction_time"])
  latency_avg = {model: np.mean(times) for model, times in latency_data.items()}

  plt.figure(figsize=(10, 5))
  plt.bar(latency_avg.keys(), latency_avg.values(), color='skyblue')
  plt.title("Average Prediction Latency by Model")
  plt.ylabel("Time (seconds)")
  plt.xticks(rotation=45)
  plt.grid(axis='y')
  plt.tight_layout()
  plt.savefig(f"{CHARTS_DIR}/predict_latency_chart.png")
  print("✅ Saved latency chart to: predict_latency_chart.png")
  plt.close()

def plot_confidence(results, headline, label_list=LABELS):
    # Filter relevant results
    filtered = [r for r in results if r["headline"] == headline]

    if not filtered:
        print(f"No results found for headline: {headline}")
        return

    models = [r["model"] for r in filtered]
    softmax_scores = [r["softmax_scores"] for r in filtered]

    indices = np.arange(len(label_list))
    width = 0.15  # Width of each bar

    plt.figure(figsize=(14, 6))

    colors = ['royalblue', 'orange', 'green', 'red', 'purple']

    for i, (model, scores) in enumerate(zip(models, softmax_scores)):
        plt.bar(indices + (i - 2) * width, scores, width=width, label=model, color=colors[i % len(colors)])

    plt.xticks(indices, label_list, rotation=45)
    plt.ylabel("Confidence Score")
    plt.title(f"Per-Label Confidence Distribution\n\"{headline}\"")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    # Safe filename
    safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', headline.strip().lower())[:20]
    filename = f"{CHARTS_DIR}/predict_confidence_{safe_title}.png"
    plt.savefig(filename)
    print(f"✅ Saved full confidence comparison chart: predict_confidence_{filename}.png")
    plt.close()

# Main
results = []
for text, true_label in TEXTS:
  for model_name, func in [
    ("Original LoRA", predict_lora),
    ("PyTorch Quantised", predict_torch),
    ("Optimum ONNX Quantised", predict_onnx_optimum),
    ("TensorRT ONNX Quantised", predict_onnx_tensorrt),
    ("TensorRT FP16", predict_trt)
  ]:
    start = time.time()
    logits = func(text)
    end = time.time()

    probs = softmax(logits)
    top5 = probs.argsort()[::-1][:5]
    result = {
      "headline": text,
      "model": model_name,
      "true_label": true_label,
      "predicted_label": LABELS[top5[0]],
      "prediction_time": round(end - start, 4),
      "softmax_scores": probs.tolist(),
    }
    for i, idx in enumerate(top5):
      label = LABELS[idx]
      score = round(probs[idx], 4)
      result[f"predict_{i+1}"] = f"{label} ({score:.4f})"
    results.append(result)

# Display as table
print("\nComparative Prediction Results:")
print(tabulate(results, headers="keys", tablefmt="github"))

# Save charts
plot_latency(results)
for text, _ in TEXTS:
    plot_confidence(results, text, LABELS)

# Clean up
cuda.Context.pop()