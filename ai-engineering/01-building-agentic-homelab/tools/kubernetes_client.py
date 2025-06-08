from kubernetes import client, config

config.load_kube_config()

def run_kubernetes_query(kind, namespace=None, filter=None):
  v1 = client.CoreV1Api()
  apps_v1 = client.AppsV1Api()
  networking_v1 = client.NetworkingV1Api()

  if kind == "pods":
    pods = v1.list_namespaced_pod(namespace=namespace) if namespace else v1.list_pod_for_all_namespaces()
    if filter and "node" in filter:
      return [p.metadata.name for p in pods.items if p.spec.node_name == filter["node"]]
    return [p.metadata.name for p in pods.items]

  if kind == "services":
    svcs = v1.list_namespaced_service(namespace=namespace) if namespace else v1.list_service_for_all_namespaces()
    return [s.metadata.name for s in svcs.items]

  if kind == "ingresses":
    ingresses = networking_v1.list_namespaced_ingress(namespace=namespace) if namespace else networking_v1.list_ingress_for_all_namespaces()
    return [i.metadata.name for i in ingresses.items]

  return {"error": f"Unsupported kind: {kind}"}
