#!/usr/bin/env sh

# inspired from https://github.com/astral-sh/uv-docker-example

if [ -t 1 ]; then
    INTERACTIVE="-it"
else
    INTERACTIVE=""
fi

podman run \
    --rm \
    --volume .:/app \
    --volume /app/.venv \
    --publish 8000:8000 \
    $INTERACTIVE \
    $(podman build -q .) \
    "$@"
