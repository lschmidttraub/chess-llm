#!/usr/bin/env bash
# Sync project files to a RunPod pod via SSH.
#
# Usage:
#   bash scripts/sync.sh <POD_ID>
#
# Requires SSH key configured in RunPod account settings.
# Uses the RunPod SSH proxy (no public IP needed).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <POD_ID>"
    echo ""
    echo "Example: $0 abc123def456"
    echo ""
    echo "Find your Pod ID with: python scripts/deploy.py status"
    exit 1
fi

POD_ID="$1"
SSH_HOST="${POD_ID}@ssh.runpod.io"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="/workspace/chess-llm"

echo "Syncing to ${SSH_HOST}:${REMOTE_DIR} ..."

rsync -avz --progress \
    -e "ssh -i ${SSH_KEY}" \
    --exclude '.venv/' \
    --exclude '.git/' \
    --exclude 'data/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude '.claude/' \
    --exclude '.flash/' \
    --exclude 'uv.lock' \
    --exclude 'notebooks/' \
    ./ "${SSH_HOST}:${REMOTE_DIR}/"

echo ""
echo "Sync complete! Connect with:"
echo "  ssh ${SSH_HOST} -i ${SSH_KEY}"
echo ""
echo "Then run:"
echo "  cd ${REMOTE_DIR}"
echo "  python scripts/preprocess.py --source remote --max-samples 100000"
echo "  python scripts/train.py"
