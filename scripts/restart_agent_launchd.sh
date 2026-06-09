#!/bin/bash
set -euo pipefail

LABEL="com.kotarou.agent"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

if [ ! -f "$PLIST" ]; then
  echo "Missing plist: $PLIST"
  echo "Run ./scripts/install_agent_launchd.sh first."
  exit 1
fi

launchctl unload "$PLIST" >/dev/null 2>&1 || true
launchctl load "$PLIST"

echo "Restarted launchd service: ${LABEL}"
