import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import random
import csv
import tqdm

# === Configuration ===
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
device = "cuda" if torch.cuda.is_available() else "cpu"
topics = ["climate", "education", "health", "security", "tech-policy"]
samples_per_topic = 2000
temperature = 0.7
max_new_tokens = 256

# === Load Model & Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  torch_dtype=torch.float16,
  device_map="auto",
  trust_remote_code=True
)
model.generation_config = model.generation_config or model.config

# === Prompt Template ===
def make_prompt(topic):
  return f"""You are a professional UK news editor.

Generate 10 realistic and stylistically diverse UK news headlines related to the topic: {topic}.

Each headline should resemble something seen on BBC, The Guardian, or Reuters. Headlines must be between 8-12 words long.

Do not include numbered lists or introduce the topic in the format 'Topic: ...'. If the topic or a colon appears in the headline, it must sound like a natural part of the headline.
List only the headlines, one per line.
"""

# === Generate Headlines ===
def generate_headlines(prompt):
  inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
  outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=temperature,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
  )
  response = tokenizer.decode(outputs[0], skip_special_tokens=True)

  # Keep only lines that look like headlines (start with a number or quote)
  headlines = [
    line.strip("-• ").strip().lstrip("0123456789. ").strip().strip('"“”')
    for line in response.split("\n")
    if line.strip() and (
      line.strip()[0].isdigit() or line.strip().startswith('"')
    )
  ]

  return headlines

# Prepare output file
csv_path = "synthetic_news_mistral_10k.csv"
with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
  writer = csv.DictWriter(file, fieldnames=["text", "label"])
  writer.writeheader()

  for topic in topics:
    generated = set()
    print(f"\n⏳ Generating headlines for: {topic}")
    with tqdm.tqdm(total=samples_per_topic, unit="headline") as pbar:
      while len(generated) < samples_per_topic:
        headlines = generate_headlines(make_prompt(topic))
        for headline in headlines:
          cleaned = headline.strip("- ").strip()
          if len(cleaned) > 15 and cleaned not in generated:
            generated.add(cleaned)
            writer.writerow({"text": cleaned, "label": topic})
            pbar.update(1)
          if len(generated) >= samples_per_topic:
            break

print(f"\n✅ Done! Saved {samples_per_topic * len(topics)} synthetic headlines to 'synthetic_news_mixtral_10k.csv'")
