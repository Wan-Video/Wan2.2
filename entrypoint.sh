#!/usr/bin/env bash
set -euo pipefail

# If a workflow path is passed in WORKFLOW_JSON, load it
if [[ -n "${WORKFLOW_JSON:-}" ]]; then
  echo "Applying workflow from WORKFLOW_JSON"
  cp "$WORKFLOW_JSON" /workspace/workflow.json
fi

# Default to launching ComfyUI server
cd /ComfyUI
exec python main.py --listen 0.0.0.0 --port "${PORT:-8188}" "$@"
