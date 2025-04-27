import os
import onnx
import torch
import tensorrt as trt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

# Paths
MODEL_DIR_BASE = "../../04-model-optimisation"
LORA_MODEL_PATH = Path(os.path.join(MODEL_DIR_BASE, "lora/lora-news/full-model")).resolve()
ONNX_MODEL_PATH = Path(os.path.join(MODEL_DIR_BASE, "quantisation/quantised-model-tensorrt/model.onnx")).resolve()
TRT_ENGINE_PATH = Path(os.path.join(MODEL_DIR_BASE, "quantisation/quantised-model-tensorrt/model_fp16.engine")).resolve()

# 1. Check LoRA Fine-tuned PyTorch Model
print("\n=== Checking LoRA PyTorch Model ===")
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
  LORA_MODEL_PATH,
  ignore_mismatched_sizes=True,
  local_files_only=True,
  use_safetensors=True
)

print(f"Tokenizer expects inputs: {tokenizer.model_input_names}")
print(f"Model type: {model.config.model_type}")
print("⚡ Note: PyTorch models don't enforce input dtype at config level — usually handled at runtime.")

# 2. Check ONNX Model
print("\n=== Checking ONNX Model Inputs ===")
onnx_model = onnx.load(ONNX_MODEL_PATH)

for input_tensor in onnx_model.graph.input:
  name = input_tensor.name
  tensor_type = input_tensor.type.tensor_type
  elem_type = tensor_type.elem_type
  shape = [dim.dim_value for dim in tensor_type.shape.dim]
  dtype_name = onnx.TensorProto.DataType.Name(elem_type)
  print(f"[Input] {name}: {dtype_name}")

# 3. Check TensorRT Engine Bindings
print("\n=== Checking TensorRT Engine Bindings ===")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open(TRT_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)

try:
  for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    dtype = engine.get_binding_dtype(i)
    is_input = engine.binding_is_input(i)
    print(f"{'[Input]' if is_input else '[Output]'} {name}: {dtype}")
except AttributeError:
  print("\nERROR: TensorRT engine is corrupted or incomplete. Could not inspect bindings.\n")
