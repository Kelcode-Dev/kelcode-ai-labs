from pathlib import Path
from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType

# === Paths ===
model_path = str(Path("../lora/lora-news/full-model").resolve())
export_path = Path("quantised-model-optimum").resolve()
export_path.mkdir(exist_ok=True)
export_path_str = str(export_path)

# === Step 1: Export to ONNX ===
print("Exporting model to ONNX...")
main_export(
  model_name_or_path=model_path,
  output=export_path_str,
  task="text-classification",
  opset=17
)

# === Step 2: Quantise with ONNX Runtime
print("Quantising ONNX model with dynamic INT8...")
onnx_model_path = export_path / "model.onnx"
quantised_model_path = export_path / "model_quantised.onnx"

quantize_dynamic(
  model_input=str(onnx_model_path),
  model_output=str(quantised_model_path),
  weight_type=QuantType.QInt8
)

print(f"✅ Quantised ONNX model saved to: {quantised_model_path}")
