#!/bin/bash
set -e

# Persist .claude.json inside the volume so token refreshes survive restarts
if [ ! -f /root/.claude/claude.json ]; then
    # First run after login: seed from backup
    if ls /root/.claude/backups/.claude.json.backup.* 1>/dev/null 2>&1; then
        latest_backup="$(ls -t /root/.claude/backups/.claude.json.backup.* | head -1)"
        cp "$latest_backup" /root/.claude/claude.json
        echo "Seeded Claude config from backup: $(basename "$latest_backup")"
    fi
fi
# Symlink so all reads/writes to .claude.json go through the volume
ln -sf /root/.claude/claude.json /root/.claude.json

if [ "$1" = "login" ]; then
    exec claude login
fi

if ! claude auth status > /dev/null 2>&1; then
    echo "ERROR: Claude is not authenticated."
    echo "Run this container with 'login' first:"
    echo "  docker run -it -v claude-auth:/root/.claude <image> login"
    exit 1
fi

exec "$@"
