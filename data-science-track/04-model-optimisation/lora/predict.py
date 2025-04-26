import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os
import sys
import torch.nn.functional as F

# === Setup paths ===
base_model_path = "lora-news/full-model"
adapter_path = "lora-news/adapter"

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForSequenceClassification.from_pretrained(
  base_model_path,
  ignore_mismatched_sizes=True,
  local_files_only=True,
  use_safetensors=True
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# === Predict function ===
def predict(text: str):
  inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
  with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1).squeeze()

    # Top-5 predictions
    topk = torch.topk(probs, k=5)
    topk_probs = topk.values.tolist()
    topk_indices = topk.indices.tolist()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    topk_labels = [id2label[i] for i in topk_indices]

    return list(zip(topk_labels, topk_probs))

# === CLI Entry Point ===
if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python predict.py \"Your input text here\"")
    sys.exit(1)

  input_text = " ".join(sys.argv[1:])
  predictions = predict(input_text)

  print("\n📢 Top 5 Predictions:")
  for label, prob in predictions:
    print(f"  - {label:<12} {prob:.4f}")
