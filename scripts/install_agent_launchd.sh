#!/bin/bash
set -euo pipefail

ROOT="/Users/Apple/Downloads/kotarou-agent-study-main"
LABEL="com.kotarou.agent"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

mkdir -p "$ROOT/logs"
mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${ROOT}/scripts/run_telegram_agent.sh</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${ROOT}</string>

  <key>StandardOutPath</key>
  <string>${ROOT}/logs/agent.out.log</string>

  <key>StandardErrorPath</key>
  <string>${ROOT}/logs/agent.err.log</string>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>
</dict>
</plist>
PLIST

launchctl unload "$PLIST" >/dev/null 2>&1 || true
launchctl load "$PLIST"

echo "Installed launchd service: ${LABEL}"
echo ""
echo "安装："
echo "./scripts/install_agent_launchd.sh"
echo "查看是否运行："
echo "launchctl list | grep com.kotarou.agent"
echo "查看日志："
echo "tail -f logs/agent.out.log"
echo "tail -f logs/agent.err.log"
echo "重启："
echo "./scripts/restart_agent_launchd.sh"
echo "卸载："
echo "./scripts/uninstall_agent_launchd.sh"
