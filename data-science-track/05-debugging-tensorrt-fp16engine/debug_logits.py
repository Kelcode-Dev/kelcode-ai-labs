import os
import sys
import torch
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# config
TEXT = "Cybersecurity breach exposes NHS patient records"
LABELS = ['World', 'Sports', 'Business', 'Sci/Tech', 'climate', 'education', 'health', 'security', 'tech-policy']
TOKENIZER_PATH = "bert-base-uncased"
MODEL_DIR_BASE = "../../04-model-optimisation/quantisation"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# load tokeniser
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Load PyTorch Model
def load_torch_model(model_dir):
  model_path = os.path.join(model_dir, "quantised_model.pt")
  model = torch.load(model_path, map_location="cpu", weights_only=False)
  model.eval()
  return model

# Load ONNX Model
def load_onnx_model(model_dir, model_name):
  model_path = os.path.join(model_dir, model_name)
  session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
  return session

# Load TensorRT Engine
def load_tensorrt_model(model_dir):
  model_path = os.path.join(model_dir, "model_fp16.engine")
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  with open(model_path, "rb") as f:
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
  return engine

# Run Inference
def predict_pytorch(model, text):
  inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
  with torch.no_grad():
    logits = model(**inputs).logits.squeeze().cpu().numpy()
  return logits

def predict_onnx(session, text):
  inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
  input_ids = inputs["input_ids"].astype(np.int64)
  attention_mask = inputs["attention_mask"].astype(np.int64)
  token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
  ort_inputs = {
    session.get_inputs()[0].name: input_ids,
    session.get_inputs()[1].name: attention_mask,
    session.get_inputs()[2].name: token_type_ids
  }
  logits = session.run(None, ort_inputs)[0].squeeze()
  return logits

def predict_tensorrt(engine, text):
  context = engine.create_execution_context()
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

# Load Models
torch_model = load_torch_model(os.path.join(MODEL_DIR_BASE, "quantised-model-torch"))
onnx_session = load_onnx_model(os.path.join(MODEL_DIR_BASE, "quantised-model-optimum"), "model_quantised.onnx")
tensorrt_onnx_session = load_onnx_model(os.path.join(MODEL_DIR_BASE, "quantised-model-tensorrt"), "model.onnx")
tensorrt_engine = load_tensorrt_model(os.path.join(MODEL_DIR_BASE, "quantised-model-tensorrt"))

# Softmax
def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def top_preds(logits, label_list):
  probs = softmax(logits)
  top_indices = probs.argsort()[::-1][:5]
  return [(label_list[i], round(probs[i], 4)) for i in top_indices]

# Inference & Print
print("\n[PyTorch Quantised]")
torch_logits = predict_pytorch(torch_model, TEXT)
print("Logits:", np.round(torch_logits, 4))
print("Top Predictions:", top_preds(torch_logits, LABELS))

print("\n[ONNX Quantised]")
onnx_logits = predict_onnx(onnx_session, TEXT)
print("Logits:", np.round(onnx_logits, 4))
print("Top Predictions:", top_preds(onnx_logits, LABELS))

print("\n[ONNX Quantised for TensorRT]")
tensorrt_onnx_logits = predict_onnx(tensorrt_onnx_session, TEXT)
print("Logits:", np.round(tensorrt_onnx_logits, 4))
print("Top Predictions:", top_preds(tensorrt_onnx_logits, LABELS))

print("\n[TensorRT FP16]")
tensorrt_logits = predict_tensorrt(tensorrt_engine, TEXT)
print("Logits:", np.round(tensorrt_logits, 4))
print("Top Predictions:", top_preds(tensorrt_logits, LABELS))

# Plotting: Softmax Confidence
plt.figure(figsize=(12, 4))
for i, (logits, title) in enumerate(zip([
  torch_logits, onnx_logits, tensorrt_onnx_logits, tensorrt_logits
], ["PyTorch", "ONNX", "Tensorrt ONNX", "TensorRT"])):
  plt.subplot(1, 4, i+1)
  plt.bar(range(len(LABELS)), softmax(logits))
  plt.xticks(range(len(LABELS)), LABELS, rotation=90)
  plt.title(title + " Softmax")
  plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/softmax_confidence_comparison.png")
print(">> Saved softmax confidence comparison to: softmax_confidence_comparison.png")

# Plotting: Raw Logits Line Chart
plt.figure(figsize=(10, 5))
plt.plot(LABELS, torch_logits, label="PyTorch", marker='o')
plt.plot(LABELS, onnx_logits, label="ONNX", marker='s')
plt.plot(LABELS, tensorrt_onnx_logits, label="Tensorrt ONNX", marker='+')
plt.plot(LABELS, tensorrt_logits, label="TensorRT", marker='^')
plt.title("Raw Logits by Backend")
plt.ylabel("Logit Value")
plt.xlabel("Labels")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/logits_line_chart.png")
print(">> Saved raw logits line chart to: logits_line_chart.png")
