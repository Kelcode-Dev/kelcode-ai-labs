import json
import random
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

random.seed(42)

CATEGORIES = {
  "Networking": [
    "Setting up VLANs for IoT devices",
    "Running Pi-hole as a DNS filter",
    "Enabling IPv6 on Ubiquiti gear",
    "Dynamic DHCP with VLAN tagging",
    "Tailscale on multiple subnets",
    "OpenWRT VLAN trunk config"
  ],
  "Storage": [
    "Building a ZFS NAS on Debian",
    "Backing up with Restic to S3",
    "Mounting NFS volumes on boot",
    "Syncing with rsync over SSH",
    "RAID vs ZFS for home servers"
  ],
  "Virtualisation": [
    "Installing Proxmox on a Mini PC",
    "Using PCI passthrough for GPU",
    "Cloning VM templates in Proxmox",
    "Live migration between nodes",
    "KVM network bridge setup"
  ],
  "Monitoring": [
    "Grafana and Prometheus setup",
    "Docker logging with Loki",
    "Self-hosting Uptime Kuma",
    "Systemd service watchdogs",
    "Detecting and rebooting crashed VMs"
  ],
  "Self-hosting": [
    "Running Nextcloud in Docker",
    "Installing Home Assistant on Pi",
    "Self-hosting Paperless-ngx",
    "Using Gitea for private Git",
    "Authelia SSO for self-hosted apps"
  ],
  "Automation": [
    "Ansible playbooks for updates",
    "Cron jobs for system health checks",
    "SSH scripts for daily snapshots",
    "Systemd timers for service rotation",
    "Wake-on-LAN automation"
  ]
}

HOMELAB_APPS = [
  "Minikube", "k3s", "k0s", "MicroK8s", "Docker Compose", "Portainer", "Pi-hole",
  "AdGuard Home", "Unbound", "Tailscale", "WireGuard", "Caddy", "Nginx Proxy Manager",
  "Home Assistant", "ESPHome", "Node-RED", "MQTT", "InfluxDB", "Grafana", "Prometheus",
  "Netdata", "Glances", "Traefik", "Watchtower", "Ollama", "Gitea", "Authelia",
  "Ansible", "Terraform", "Pulumi", "Speedtest Tracker", "Uptime Kuma", "Paperless-ngx",
  "Jellyfin", "Immich", "Photoprism", "Navidrome", "Audiobookshelf", "Tandoor Recipes",
  "Mealie", "Vaultwarden", "Bitwarden_RS", "Nextcloud", "LibreSpeed", "Calibre Web",
  "Trilium Notes", "Docspell", "Outline", "Homer", "Heimdall", "FileBrowser", "Shlink",
  "Freshrss", "Miniflux", "Ntfy", "Mattermost", "Jitsi", "Snipe-IT", "Docker Mailserver",
  "Cloudflare Tunnel", "Whoogle", "Headscale", "Netbird", "Zitadel", "ESP32 OTA Server",
  "Postgres", "MySQL", "MongoDB", "Redis", "InfluxDB"
]

def generate_homelab_app_titles(n=70):
  return [f"Installing {app} in your homelab" for app in random.sample(HOMELAB_APPS, n)]

def generate_content(title, category):
  sections = [
    f"## Overview\nThis guide walks through how to set up **{title}** in a homelab.\n",
    "## Installation Steps\n1. Prepare your environment\n2. Pull Docker image or install package\n3. Configure ports and volumes\n",
    "## Configuration Tips\n- Set up secure credentials\n- Expose only necessary ports\n- Use persistent volumes\n",
    f"```bash\n# Example command\ndocker run --name app -v /data:/data -p 8080:80 some-image\n```\n",
    "_Last updated: {}_\n".format(datetime.now(timezone.utc).date())
  ]
  random.shuffle(sections)
  return f"# {title}\n\n" + "\n".join(sections)

def generate_docs():
  docs = []

  for category, titles in CATEGORIES.items():
    for title in titles:
      docs.append({
        "title": title,
        "category": category,
        "tags": [category.lower()] + title.lower().split()[:2],
        "content": generate_content(title, category),
        "source": f"{category.lower()}-guide.md"
      })

  app_titles = generate_homelab_app_titles(n=70)
  for title in app_titles:
    docs.append({
      "title": title,
      "category": "HomelabApps",
      "tags": ["homelabapps"] + title.lower().split()[:2],
      "content": generate_content(title, "HomelabApps"),
      "source": "apps-guide.md"
    })

  return sorted(docs, key=lambda d: d["title"])

def main():
  out_path = Path("seed_documents/seed_docs.jsonl")
  out_path.parent.mkdir(parents=True, exist_ok=True)

  print("Generating homelab + app install documents...\n")
  docs = generate_docs()
  with out_path.open("w", encoding="utf-8") as f:
    for doc in tqdm(docs, desc="📦 Writing documents"):
      f.write(json.dumps(doc) + "\n")
  print(f"\nDone! Generated {len(docs)} documents → {out_path}")

if __name__ == "__main__":
  main()
