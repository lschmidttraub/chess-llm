"""Manage RunPod GPU pods for chess-llm training.

Usage:
    python scripts/deploy.py create [--gpu GPU_TYPE]
    python scripts/deploy.py status
    python scripts/deploy.py stop POD_ID
    python scripts/deploy.py terminate POD_ID
    python scripts/deploy.py list-gpus
"""

import argparse
import json
import os
import sys

import runpod
from dotenv import load_dotenv


DEFAULT_GPU = "NVIDIA RTX 3090"
DEFAULT_VOLUME_GB = 100
DEFAULT_CONTAINER_DISK_GB = 50
DOCKER_IMAGE = "leosct/chess-llm:latest"
POD_NAME = "chess-llm-training"


def init_api():
    load_dotenv()
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY not found in environment or .env file")
        sys.exit(1)
    runpod.api_key = api_key


def create_pod(gpu_type: str, volume_gb: int):
    """Create a new GPU pod with the unsloth Docker image."""
    print(f"Creating pod '{POD_NAME}' with {gpu_type}...")

    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=DOCKER_IMAGE,
        gpu_type_id=gpu_type,
        volume_in_gb=volume_gb,
        container_disk_in_gb=DEFAULT_CONTAINER_DISK_GB,
        ports="8888/http,22/tcp",
        support_public_ip=True,
        start_ssh=True,
        env={
            "JUPYTER_PASSWORD": "chess-llm",
        },
    )

    print(f"\nPod created successfully!")
    print(f"  Pod ID: {pod['id']}")
    print(f"  Status: {pod.get('desiredStatus', 'PENDING')}")
    print(f"\nConnect via SSH once ready:")
    print(f"  ssh {pod['id']}@ssh.runpod.io -i ~/.ssh/id_ed25519")
    print(f"\nOr check the RunPod dashboard for the full SSH command with public IP.")
    return pod


def list_pods():
    """List all pods and their status."""
    pods = runpod.get_pods()
    if not pods:
        print("No pods found.")
        return

    for pod in pods:
        gpu = pod.get("machine", {}).get("gpuDisplayName", "N/A")
        print(f"  {pod['id']}  {pod.get('name', 'N/A'):30s}  {pod.get('desiredStatus', 'N/A'):10s}  {gpu}")


def stop_pod(pod_id: str):
    """Stop a running pod (preserves volume data)."""
    print(f"Stopping pod {pod_id}...")
    runpod.stop_pod(pod_id)
    print("Pod stopped. Volume data is preserved. Use 'resume' to restart.")


def terminate_pod(pod_id: str):
    """Terminate a pod (deletes everything)."""
    print(f"Terminating pod {pod_id}...")
    runpod.terminate_pod(pod_id)
    print("Pod terminated.")


def list_gpus():
    """List available GPU types and pricing."""
    gpus = runpod.get_gpus()
    print(f"{'GPU Type':<40s} {'VRAM':>6s} {'Secure Cloud':>14s} {'Community':>12s}")
    print("-" * 75)
    for gpu in sorted(gpus, key=lambda g: g.get("memoryInGb", 0)):
        name = gpu.get("id", "N/A")
        vram = gpu.get("memoryInGb", 0)
        secure = gpu.get("securePrice", None)
        community = gpu.get("communityPrice", None)
        secure_str = f"${secure:.2f}/hr" if secure else "N/A"
        community_str = f"${community:.2f}/hr" if community else "N/A"
        print(f"  {name:<38s} {vram:>4d}GB {secure_str:>14s} {community_str:>12s}")


def main():
    parser = argparse.ArgumentParser(description="Manage RunPod pods for training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    create_parser = subparsers.add_parser("create", help="Create a new GPU pod")
    create_parser.add_argument("--gpu", default=DEFAULT_GPU, help=f"GPU type (default: {DEFAULT_GPU})")
    create_parser.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB, help="Volume size in GB")

    # status
    subparsers.add_parser("status", help="List all pods")

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop a pod")
    stop_parser.add_argument("pod_id", help="Pod ID to stop")

    # terminate
    term_parser = subparsers.add_parser("terminate", help="Terminate a pod")
    term_parser.add_argument("pod_id", help="Pod ID to terminate")

    # list-gpus
    subparsers.add_parser("list-gpus", help="List available GPU types")

    args = parser.parse_args()
    init_api()

    if args.command == "create":
        create_pod(args.gpu, args.volume_gb)
    elif args.command == "status":
        list_pods()
    elif args.command == "stop":
        stop_pod(args.pod_id)
    elif args.command == "terminate":
        terminate_pod(args.pod_id)
    elif args.command == "list-gpus":
        list_gpus()


if __name__ == "__main__":
    main()
