#!/bin/bash
set -euo pipefail

LABEL="com.kotarou.agent"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

if [ -f "$PLIST" ]; then
  launchctl unload "$PLIST" >/dev/null 2>&1 || true
  rm -f "$PLIST"
  echo "Uninstalled launchd service: ${LABEL}"
else
  launchctl remove "$LABEL" >/dev/null 2>&1 || true
  echo "No plist found for ${LABEL}; nothing to delete."
fi

echo "Kept .env, session files, and logs."
