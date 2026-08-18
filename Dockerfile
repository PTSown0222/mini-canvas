FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /src

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# copy source and sync
COPY . /src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable


COPY --from=builder /app/.venv /app/.venv
COPY . /src

CMD ["uv", "run", "my_app"]
