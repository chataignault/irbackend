
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
RUN uv tool install harlequin
ENV PATH="/root/.local/bin:$PATH"
CMD ["harlequin"]

