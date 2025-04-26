import torch
import torch.quantization
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import os

# === Paths ===
base_model_path = Path("../lora/lora-news/full-model").resolve()  # Base model (for tokenizer)
quantised_output_path = Path("quantised-model-torch").resolve()  # Quantised model path
quantised_output_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

# === Load model and tokenizer ===
print("Loading model and tokenizer from:", base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForSequenceClassification.from_pretrained(
  base_model_path,
  torch_dtype=torch.float32
)

# === Apply dynamic quantisation ===
print("Quantising with PyTorch dynamic quantisation (INT8)...")
quantised_model = torch.quantization.quantize_dynamic(
  model,
  {torch.nn.Linear},  # Apply quantisation to Linear layers
  dtype=torch.qint8  # Use INT8 precision
)

# === Save quantised model manually (not using save_pretrained) ===
output_dir = "quantised-model-torch"
quantised_model.config.save_pretrained(quantised_output_path)
torch.save(quantised_model, quantised_output_path / "quantised_model.pt")
tokenizer.save_pretrained(quantised_output_path)

print(f"✅ Quantised model saved to: {quantised_output_path}")
