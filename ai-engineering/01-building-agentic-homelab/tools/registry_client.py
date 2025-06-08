import os
import requests
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Your Harbor instance details
HARBOR_URL = os.getenv("HARBOR_URL")
HARBOR_USER = os.getenv("HARBOR_USERNAME")
HARBOR_TOKEN = os.getenv("HARBOR_TOKEN")

def query_harbor_registry(operation: str, project_name: str = None, repository_name: str = None) -> Dict[str, Any]:
  """
  Queries the Harbor registry for information using Basic Authentication.

  Args:
    operation (str): The action to perform. Supported: 'list_repositories', 'list_tags'.
    project_name (str, optional): The name of the project.
    repository_name (str, optional): The name of the repository (don't include project name).

  Returns:
    Dict[str, Any]: A dictionary containing the result or an error.
  """
  if not HARBOR_USER or not HARBOR_TOKEN:
    return {"error": "HARBOR_USERNAME and/or HARBOR_TOKEN environment variables not set."}

  # The requests library handles the Base64 encoding for Basic Auth automatically
  # when you pass the `auth` parameter as a (username, password) tuple.
  auth_credentials = (HARBOR_USER, HARBOR_TOKEN)

  # Use v2 of the Harbor API
  api_base = f"{HARBOR_URL}/api/v2.0"

  try:
    if operation == "list_repositories":
      if not project_name:
        return {"error": "Operation 'list_repositories' requires a 'project_name'."}

      url = f"{api_base}/projects/{project_name}/repositories"
      # Pass the credentials to the `auth` parameter
      response = requests.get(url, auth=auth_credentials)
      response.raise_for_status()

      repositories = response.json()
      return {"repositories": [r['name'].replace(f"{project_name}/", "") for r in repositories]}
    elif operation == "list_tags":
      if not project_name or not repository_name:
        return {"error": "Operation 'list_tags' requires both 'project_name' and 'repository_name'."}

      url = f"{api_base}/projects/{project_name}/repositories/{repository_name}/artifacts"
      # Pass the credentials to the `auth` parameter
      response = requests.get(url, auth=auth_credentials)
      response.raise_for_status()

      artifacts = response.json()
      all_tags = []
      for artifact in artifacts:
        if 'tags' in artifact and artifact['tags']:
          all_tags.extend([tag['name'] for tag in artifact['tags']])
      return {"tags": all_tags}
    else:
      return {"error": f"Unsupported Harbor operation: {operation}"}

  except requests.exceptions.HTTPError as e:
    return {"error": f"HTTP Error: {e.response.status_code} - {e.response.text}"}
  except requests.exceptions.RequestException as e:
    return {"error": f"Request failed: {e}"}
