# Galaxy-JEPA — dev container
# Python 3.11 with uv for dependency management (matches the v1 conventions,
# bumped from 3.10 to 3.11; see DECISIONS.md D3).

FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

# Bring in uv as a single static binary from the official image.
# Pinned for reproducible builds — bump the tag to update uv.
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

# UV_LINK_MODE=copy        — avoid hard-link warnings on the bind-mounted workspace
# UV_PYTHON_DOWNLOADS=never — use the image's Python 3.11 rather than a uv download
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
