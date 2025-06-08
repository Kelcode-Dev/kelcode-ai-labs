import os
import requests
from dotenv import load_dotenv

load_dotenv()

LLM_BASE = os.getenv("LLM_API_BASE")
LLM_MODEL = os.getenv("LLM_MODEL_NAME")
LLM_API_KEY = os.getenv("LLM_API_KEY", "none")

def ask_llm(question: str, data: dict) -> str:
  system_prompt = "You are a DevOps assistant. Based on the provided monitoring data, answer the user's question clearly and helpfully."

  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Question: {question}"},
    {"role": "user", "content": f"Monitoring Data:\n{data}"}
  ]

  response = requests.post(
    f"{LLM_BASE}/chat/completions",
    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
    json={
      "model": LLM_MODEL,
      "messages": messages,
      "temperature": 0.2,
      "max_tokens": 2048
    }
  )
  response.raise_for_status()
  return response.json()["choices"][0]["message"]["content"]
