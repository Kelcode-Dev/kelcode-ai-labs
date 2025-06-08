import os
import requests
from dotenv import load_dotenv

load_dotenv()
PROM_URL = os.getenv("PROMETHEUS_URL")

def run_promql_query(query):
  try:
    response = requests.get(
      f"{PROM_URL}/api/v1/query",
      params={"query": query}
    )
    response.raise_for_status()
    data = response.json()
    if data['status'] == 'success':
      return data['data']['result']
    else:
      return {"error": data.get("error", "Unknown error")}
  except requests.exceptions.HTTPError as e:
    return {"error": f"HTTP Error {response.status_code}: {response.text}"}
  except Exception as e:
    return {"error": str(e)}

if __name__ == "__main__":
  q = input("Enter PromQL: ")
  print(run_promql_query(q))
