import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import subprocess

# === Paths ===
model_dir = Path("../lora/lora-news/full-model").resolve()
onnx_path = Path("quantised-model-tensorrt/model.onnx").resolve()
engine_path = Path("quantised-model-tensorrt/model_fp16.engine").resolve()
onnx_path.parent.mkdir(parents=True, exist_ok=True)

# === Step 1: Load model and tokenizer
print("Loading 9-label model and exporting to ONNX...")

model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval()

# Dummy input
inputs = tokenizer("Example headline for export", return_tensors="pt", padding="max_length", truncation=True, max_length=128)

# Export to ONNX
torch.onnx.export(
    model,
    args=(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
    f=onnx_path.as_posix(),
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "token_type_ids": {0: "batch"}},
    opset_version=17
)

print(f"✅ ONNX model exported to: {onnx_path}")

# === Step 2: Build TensorRT engine
print("Building FP16 TensorRT engine from ONNX...")

trtexec_cmd = [
    "trtexec",
    f"--onnx={onnx_path}",
    f"--saveEngine={engine_path}",
    "--fp16",
    "--minShapes=input_ids:1x128,attention_mask:1x128,token_type_ids:1x128",
    "--optShapes=input_ids:1x128,attention_mask:1x128,token_type_ids:1x128",
    "--maxShapes=input_ids:1x128,attention_mask:1x128,token_type_ids:1x128",
]

result = subprocess.run(trtexec_cmd, capture_output=True, text=True)

print("🖨️ trtexec stdout:\n", result.stdout)
print("🖨️ trtexec stderr:\n", result.stderr)

if engine_path.exists():
    print(f"✅ TensorRT engine created at: {engine_path}")
else:
    print("❌ TensorRT engine was NOT created. Check the trtexec output above carefully!")
