import re
import json
from llm.ollama import ask_ollama_tool_call
from typing import Tuple

VALID_TOOL_ARGS = {
  "query_prometheus": {"query"},
  "query_kubernetes": {"kind", "namespace", "filter"},
  "query_harbor_registry": {"operation", "project_name", "repository_name"}
}

def decide_tool(question: str) -> tuple[str, dict]:
  routing_prompt = f"""
You are an infrastructure automation agent.

Given a user question, decide which tool to use and return the tool call in the following exact JSON format:

```
{{
"tool": "\<tool\_name>",
"args": "...",
"filter": {{...}}  # optional, can be omitted if not needed
}}

```

Only return the JSON. No explanation, no preamble.

Available tools:

1. query_prometheus
   - args: a valid PromQL query string
   - Example:
     {{
       "tool": "query_prometheus",
       "args": "avg(rate(node_cpu_seconds_total[5m]))"
     }}

2. query_kubernetes
   - Use this tool to get information about Kubernetes resources like pods, services, and ingresses.
   - **DO NOT** generate a kubectl command string.
   - Instead, provide a JSON object with the 'kind', 'namespace', and 'filter' keys.
   - Supported 'kind' values are: "pods", "services", "ingresses".
   - 'namespace' is optional.
   - 'filter' is optional and can be used for more specific queries.
   - Example 1 (all pods in a namespace):
     {{
       "tool": "query_kubernetes",
       "args": {{
         "kind": "pods",
         "namespace": "kube-system"
       }}
     }}
   - Example 2 (all pods on a specific node):
     {{
       "tool": "query_kubernetes",
       "args": {{
         "kind": "pods",
         "filter": {{
           "node": "k8s-cpu-worker02"
         }}
       }}
     }}

3. query_harbor_registry
   - Use this tool to get information about container images from the Harbor registry.
   - Provide a JSON object with the 'operation' and other required arguments.
   - Supported 'operation' values are: "list_repositories", "list_tags".
   - Example 1 (list all image repositories in the 'library' project):
     {{
       "tool": "query_harbor_registry",
       "args": {{
         "operation": "list_repositories",
         "project_name": "library"
       }}
     }}
   - Example 2 (list all tags for the 'nginx' image in the 'proxies' project):
     {{
       "tool": "query_harbor_registry",
       "args": {{
         "operation": "list_tags",
         "project_name": "proxies",
         "repository_name": "nginx"
       }}
     }}
---

Question: {question}
"""

  response = ask_ollama_tool_call(routing_prompt)
  return parse_tool_call(response)

# This function parses the tool call JSON string and validates its structure.
def parse_tool_call(tool_call: str) -> Tuple[str, dict]:
    """
    Parses a tool call string that may or may not be wrapped in markdown fences.
    """
    json_string = None

    # 1. First, try to find a JSON block wrapped in ```json ... ```
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", tool_call, re.DOTALL)

    if json_match:
        json_string = json_match.group(1).strip()
    else:
        # 2. If no markdown block, fall back to finding the first and last curly brace
        try:
            start_index = tool_call.find('{')
            end_index = tool_call.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_string = tool_call[start_index : end_index + 1]
        except Exception:
            # Pass if we can't find braces, we'll handle it next
            pass

    if not json_string:
        print(f"Could not extract a JSON object from the tool call: {tool_call}")
        return "none", {}

    # 3. Now, try to parse the extracted string
    try:
        data = json.loads(json_string)
        print("Successfully parsed JSON data:", data)

        tool = data.get("tool")
        if not tool:
            return "none", {}

        # Consolidate arguments
        final_args = {}
        args_content = data.get("args")
        if isinstance(args_content, str):
            final_args["command"] = args_content
        elif isinstance(args_content, dict):
            final_args.update(args_content)

        if "filter" in data and isinstance(data["filter"], dict):
            final_args.update(data["filter"])

        return tool, final_args

    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON string: {json_string}")
        print(f"Error: {e}")
        return "none", {}
