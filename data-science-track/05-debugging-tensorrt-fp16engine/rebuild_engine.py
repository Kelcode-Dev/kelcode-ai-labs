import torch
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Config
MODEL_DIR = Path("../../04-model-optimisation/lora/lora-news/full-model").resolve()
ONNX_PATH = Path("requantised-tensorrt/model.onnx").resolve()
ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
ENGINE_PATH = Path("requantised-tensorrt/model_fp16.engine").resolve()
TEXT = "Cybersecurity breach exposes NHS patient records"
LABELS = ['World', 'Sports', 'Business', 'Sci/Tech', 'climate', 'education', 'health', 'security', 'tech-policy']
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
cuda.Device(0).make_context()

# Step 1: Load model and tokenizer
def export_to_onnx():
  print("Exporting model to ONNX...")
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
  model.eval()

  # Dummy input
  inputs = tokenizer("Example headline for export", return_tensors="pt", padding="max_length", truncation=True, max_length=128)
  input_ids = inputs["input_ids"].to(torch.int32)
  attention_mask = inputs["attention_mask"].to(torch.int32)
  token_type_ids = inputs["token_type_ids"].to(torch.int32)

  # Export to ONNX
  torch.onnx.export(
    model,
    args=(input_ids, attention_mask, token_type_ids),
    f=ONNX_PATH.as_posix(),
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "token_type_ids": {0: "batch"}},
    opset_version=17
  )

  print(f"✅ ONNX model exported to: {ONNX_PATH}")

  onnx_logits = validate_onnx_logits(ONNX_PATH, tokenizer, TEXT, LABELS)
  return onnx_logits

# Step 2 validate the onnx model
def validate_onnx_logits(onnx_path, tokenizer, text, expected_labels):
  print("\nValidating ONNX model logits...")
  session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

  inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
  input_ids = inputs["input_ids"].astype(np.int32)
  attention_mask = inputs["attention_mask"].astype(np.int32)
  token_type_ids = np.zeros_like(input_ids, dtype=np.int32)

  ort_inputs = {
    session.get_inputs()[0].name: input_ids,
    session.get_inputs()[1].name: attention_mask,
    session.get_inputs()[2].name: token_type_ids
  }

  logits = session.run(None, ort_inputs)[0].squeeze()
  probs = softmax(logits)
  top = probs.argsort()[::-1][:5]

  return logits


# Step 3 build TensorRT engine
def build_engine():
  print("\nBuilding TensorRT engine...")
  builder = trt.Builder(TRT_LOGGER)
  network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
  network = builder.create_network(network_flags)
  parser = trt.OnnxParser(network, TRT_LOGGER)
  config = builder.create_builder_config()
  config.set_flag(trt.BuilderFlag.FP16)
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

  with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
      for i in range(parser.num_errors):
        print(parser.get_error(i))
      raise RuntimeError("Failed to parse ONNX")

  profile = builder.create_optimization_profile()
  for name in ["input_ids", "attention_mask", "token_type_ids"]:
    profile.set_shape(name, (1, 128), (1, 128), (1, 128))
  config.add_optimization_profile(profile)

  engine = builder.build_serialized_network(network, config)
  with open(ENGINE_PATH, "wb") as f:
    f.write(engine)
  print(f"✅ Engine saved to {ENGINE_PATH}")

# Run inference
def run_inference():
  tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
  with open(ENGINE_PATH, "rb") as f:
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

  context = engine.create_execution_context()
  inputs = tokenizer(TEXT, return_tensors="np", padding="max_length", truncation=True, max_length=128)
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

# Softmax and top-k
def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def top_preds(logits, labels):
  probs = softmax(logits)
  top = probs.argsort()[::-1][:5]
  return [(labels[i], round(probs[i], 4)) for i in top]

# Main
if __name__ == "__main__":
  onnx_logits = export_to_onnx()
  build_engine()
  tensorrt_logits = run_inference()

  print("\n[Onnx Inference Result]")
  print("ONNX Logits:", np.round(onnx_logits, 4))
  print("Top Predictions:", top_preds(onnx_logits, LABELS))

  print("\n[TensorRT FP16 Inference Result]")
  print("Logits:", np.round(tensorrt_logits, 4))
  print("Top Predictions:", top_preds(tensorrt_logits, LABELS))

  plt.figure(figsize=(12, 4))
  for i, (logits, title) in enumerate(zip([
    onnx_logits, tensorrt_logits
  ], ["ONNX", "TensorRT"])):
    plt.subplot(1, 2, i+1)
    plt.bar(range(len(LABELS)), softmax(logits))
    plt.xticks(range(len(LABELS)), LABELS, rotation=90)
    plt.title(title + " Softmax")
    plt.ylim(0, 1)
  plt.tight_layout()
  plt.savefig("rebuild_softmax_confidence_comparison.png")
  print(">> Saved softmax confidence comparison to: softmax_confidence_comparison.png")

  # Plotting: Raw Logits Line Chart
  plt.figure(figsize=(10, 5))
  plt.plot(LABELS, onnx_logits, label="ONNX", marker='s')
  plt.plot(LABELS, tensorrt_logits, label="TensorRT", marker='^')
  plt.title("Raw Logits by Backend")
  plt.ylabel("Logit Value")
  plt.xlabel("Labels")
  plt.xticks(rotation=45)
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig("rebuild_logits_line_chart.png")
  print(">> Saved raw logits line chart to: logits_line_chart.png")


# Clean up
cuda.Context.pop()