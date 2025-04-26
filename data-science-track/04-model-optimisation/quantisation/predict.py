import time
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import matplotlib.pyplot as plt
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime import InferenceSession
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from transformers import AutoConfig

# Paths
onnx_model_path = "quantised-model-optimum/model_quantised.onnx"
engine_model_path = "quantised-model-tensorrt/model_fp16.engine"
quantised_model_path = "quantised-model-torch"

# Initialize CUDA context
cuda.init()
device = cuda.Device(0)
cuda_ctx = device.make_context()

# Step 1 - Load the base model and LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(quantised_model_path)
pytorch_model_path = os.path.join(quantised_model_path, "quantised_model.pt")
pytorch_model = torch.load(pytorch_model_path, map_location="cpu", weights_only=False)
pytorch_model.eval()

# 2. Load ONNX model (CPU)
onnx_session = InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# 3. Load TensorRT Engine (GPU)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(engine_model_path, "rb") as f:
  engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
trt_context = engine.create_execution_context()

# Prediction Function: PyTorch
def predict_pytorch(text):
  inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
  with torch.no_grad():
    outputs = pytorch_model(**inputs)
    logits = outputs.logits.numpy()
    return logits

# Prediction Function: ONNX
def predict_onnx(text):
  inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
  input_ids = np.array(inputs['input_ids'], dtype=np.int64)
  attention_mask = np.array(inputs['attention_mask'], dtype=np.int64)
  token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

  onnx_inputs = {
    onnx_session.get_inputs()[0].name: input_ids,
    onnx_session.get_inputs()[1].name: attention_mask,
    onnx_session.get_inputs()[2].name: token_type_ids
  }
  logits = onnx_session.run(None, onnx_inputs)[0]
  return logits

# Prediction Function: TensorRT
def predict_tensorrt(text):
  inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
  input_ids = inputs["input_ids"].astype(np.int32)
  attention_mask = inputs["attention_mask"].astype(np.int32)
  token_type_ids = np.zeros_like(input_ids, dtype=np.int32)

  # Set shapes if engine supports dynamic shapes
  trt_context.set_input_shape("input_ids", input_ids.shape)
  trt_context.set_input_shape("attention_mask", attention_mask.shape)
  trt_context.set_input_shape("token_type_ids", token_type_ids.shape)

  input_data = [input_ids, attention_mask, token_type_ids]

  bindings = []
  device_buffers = []

  # Allocate device memory for inputs
  for inp in input_data:
    d_in = cuda.mem_alloc(inp.nbytes)
    cuda.memcpy_htod(d_in, inp)
    bindings.append(int(d_in))
    device_buffers.append(d_in)

  # Allocate device memory for output
  output_shape = trt_context.get_tensor_shape("logits")
  output = np.empty(output_shape, dtype=np.float32)
  d_out = cuda.mem_alloc(output.nbytes)
  bindings.append(int(d_out))
  device_buffers.append(d_out)

  # Run inference
  trt_context.execute_v2(bindings)

  # Copy prediction back
  cuda.memcpy_dtoh(output, d_out)

  return output

config = AutoConfig.from_pretrained(quantised_model_path)
id2label = {int(k): v for k, v in config.id2label.items()}
label_list = [id2label[i] for i in sorted(id2label)]

def run_comparison(text):
  print(f"\nPredicting: '{text}'\n")

  def print_top5(logits, duration, backend):
    probs = softmax(logits[0])
    top_indices = np.argsort(probs)[::-1][:5]
    print(f"[{backend}]: {id2label[np.argmax(probs)]} ({np.argmax(probs)}) | {duration:.4f} sec")
    print(f"Top 5 Predictions:")
    for i in top_indices:
      print(f"  - {id2label[i]:<12} {probs[i]:.4f}")
    print("")

  # PyTorch
  start = time.time()
  pytorch_logits = predict_pytorch(text)
  pytorch_time = time.time() - start
  print_top5(pytorch_logits, pytorch_time, "PyTorch Quantised")

  # ONNX
  start = time.time()
  onnx_logits = predict_onnx(text)
  onnx_time = time.time() - start
  print_top5(onnx_logits, onnx_time, "ONNX Quantised")

  # TensorRT
  start = time.time()
  tensorrt_logits = predict_tensorrt(text)
  tensorrt_time = time.time() - start
  print_top5(tensorrt_logits, tensorrt_time, "TensorRT FP16")

  plot_confidence_comparison(label_list, pytorch_logits, onnx_logits, tensorrt_logits, text)

def plot_confidence_comparison(label_list, pytorch_logits, onnx_logits, tensorrt_logits, text):
  pytorch_logits = softmax(np.array(pytorch_logits).flatten())
  onnx_logits = softmax(np.array(onnx_logits).flatten())
  tensorrt_logits = softmax(np.array(tensorrt_logits).flatten())

  indices = np.arange(len(label_list))
  width = 0.25

  plt.figure(figsize=(10, 5))
  plt.bar(indices - width, pytorch_logits, width=width, label='PyTorch')
  plt.bar(indices, onnx_logits, width=width, label='ONNX')
  plt.bar(indices + width, tensorrt_logits, width=width, label='TensorRT')

  plt.xticks(indices, label_list, rotation=45)
  plt.ylabel("Confidence Score")
  plt.title("Top-5 Confidence Scores by Backend")
  plt.xlabel(text)
  plt.legend()
  plt.tight_layout()

  # Clean filename
  safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', text.strip().lower())[:20]
  filename = f"confidence_{safe_title}.png"
  plt.savefig(filename)
  plt.close()
  print(f">> Saved prediction confidence comparison to: {filename}")

# Run it ...
try:
  texts = [
    "Cybersecurity breach exposes NHS patient records",
    "AI tutor program set to roll out in Scottish schools",
    "Climate report warns UK cities face extreme flooding by 2030",
  ]
  for t in texts:
    run_comparison(t)
finally:
  del trt_context
  del engine
  cuda.Context.synchronize()

  cuda_ctx.pop()
  cuda_ctx.detach()
