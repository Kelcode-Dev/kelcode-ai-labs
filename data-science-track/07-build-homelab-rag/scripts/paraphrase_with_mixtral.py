import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
from tqdm import tqdm

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
PROMPT_TEMPLATE = """<s>[INST]
You are helping rewrite technical homelab Markdown articles.

Please paraphrase the following article:
- Keep the structure and Markdown formatting (e.g. headings, steps, bold)
- Change the wording and phrasing throughout
- Add 1-2 realistic gotchas or notes
- Preserve technical accuracy

Original:
{input}

Paraphrased:
[/INST]
"""

def load_seed_docs(path):
  with open(path, "r", encoding="utf-8") as f:
    return [json.loads(line) for line in f]

def main():
  print("Loading model:", MODEL_NAME)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
  )

  paraphraser = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,  # lower this if it's too slow
    temperature=0.7
  )

  print("Loading seed documents...")
  seed_docs = load_seed_docs("seed_documents/seed_docs.jsonl")
  print(f"Starting paraphrasing of {len(seed_docs)} documents...\n")

  out_path = Path("paraphrased_documents/mixtral_paraphrased.jsonl")
  out_path.parent.mkdir(parents=True, exist_ok=True)

  with out_path.open("w", encoding="utf-8") as f:
    for doc in tqdm(seed_docs, desc="Paraphrasing docs"):
      prompt = PROMPT_TEMPLATE.format(input=doc["content"])
      try:
        result = paraphraser(prompt)[0]["generated_text"]
        doc["paraphrased_content"] = result
        f.write(json.dumps(doc) + "\n")
      except Exception as e:
        print(f"Error on {doc['title']}: {e}")

  print("\nAll documents paraphrased!")

if __name__ == "__main__":
  main()
