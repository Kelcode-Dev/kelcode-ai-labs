from fastapi import FastAPI
from pydantic import BaseModel
from tools.prometheus_client import run_promql_query
from tools.kubernetes_client import run_kubernetes_query
from tools.registry_client import query_harbor_registry
from tool_router import decide_tool
from llm.mistral import ask_llm

def balance_parentheses(promql: str) -> str:
  open_count = promql.count("(")
  close_count = promql.count(")")
  return promql + (")" * (open_count - close_count))

app = FastAPI()

class Query(BaseModel):
  question: str

@app.post("/ask")
async def ask_agent(query: Query):
  question = query.question
  tool_name, args = decide_tool(question)

  if tool_name == "query_prometheus" and args:
    result = run_promql_query(args.get("command", ""))
  elif tool_name == "query_kubernetes" and args:
    if "kind" in args:
      result = run_kubernetes_query(**args)
    else:
      result = {"error": "LLM failed to provide a 'kind' for the Kubernetes query."}
  elif tool_name == "query_harbor_registry":
    if "operation" in args:
      result = query_harbor_registry(**args)
    else:
      result = {"error": "LLM failed to provide an 'operation' for the Harbor query."}
  else:
    result = {"tool": "none", "result": "Tool not recognized or no tool needed"}

  return ask_llm(
    question=question,
    data=result
  )
