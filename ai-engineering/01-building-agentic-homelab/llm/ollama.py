import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE = os.getenv("OLLAMA_API_BASE")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_NAME")

def ask_ollama_tool_call(prompt: str) -> str:
  try:
    response = requests.post(
      f"{OLLAMA_BASE}/api/generate",
      headers={"Content-Type": "application/json"},
      json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
      }
    )
    response.raise_for_status()
    return response.json()["response"].strip()
  except Exception as e:
    return f"Error: {e}"
