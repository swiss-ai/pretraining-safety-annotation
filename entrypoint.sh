#!/bin/bash
set -e

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
