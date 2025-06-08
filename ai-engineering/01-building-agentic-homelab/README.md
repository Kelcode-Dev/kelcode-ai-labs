# Homelab AI Agent

A local-first, LLM-powered agent that lets you query your infrastructure with natural language. Ask questions like:

- "What's the average GPU temperature over the last 24 hours?"
- "Which pod in kube-system is using the most memory?"
- "Show me the latest images in my container registry."

Runs on your own hardware. No OpenAI keys, no telemetry, no fluff.

## 🧰 Features

- 🔌 Local LLM support via [Ollama](https://ollama.com/)
- 🛠️ OpenAI-style function calling for tool execution
- 📦 Built-in tools for:
  - Prometheus metric queries
  - Kubernetes pod and node info
  - Harbor registry image lookups
- 🧠 Model-agnostic: works with `phi3`, `mistral`, `llama3`, etc.
- ⚙️ Simple Python interface and testable curl endpoint

## 🗺️ Project Structure

```text
.
├── main.py                 # Entry point: handles user query → response
├── tool_router.py          # Constructs prompt, routes tool call via LLM
├── requirements.txt        # Project dependencies

├── llm/
│   ├── ollama.py           # Handles Ollama model interactions
│   └── mistral.py          # (Optional) Specific model tweaks

├── tools/
│   ├── kubernetes_client.py    # Functions for querying pod/node status
│   ├── prometheus_client.py    # PromQL query functions
│   └── registry_client.py      # Harbor registry tooling
````

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama and pull a model

```bash
ollama run phi4-mini
# or mistral, llama3, etc.
```

### 3. Run a test query

```bash
python main.py
```

You'll be prompted to enter a natural language query.

## 📡 Example Queries

| Prompt                                             | Tool Called             |
| -------------------------------------------------- | ----------------------- |
| What's the average GPU temp for the past 24 hours? | `query_prometheus`      |
| Which pod is using the most memory in kube-system? | `query_kubernetes`      |
| List the latest images in my Harbor registry       | `query_harbor_registry` |

## 🔧 Customisation

You can easily add new tools to the `tools/` folder and register them with a simple if/else in main.py (function schema coming soon).

Future plans include:

* Multi-step reasoning and agent planning
* Voice interaction via Whisper
* Chart/graph rendering from query output

## 🤖 Why This Exists

This project is part of my AI engineering journey to make infrastructure easier to reason about. It's designed to be:

* Lightweight
* Hackable
* 100% offline

Read more in [this blog post](kelcode.co.uk/building-a-homelab-agentic-ecosystem-part1/).
