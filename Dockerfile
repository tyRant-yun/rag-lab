FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels \
    && mkdir /data \
    && chown appuser:appuser /data

WORKDIR /app
USER appuser

EXPOSE 8000

CMD ["serve-api"]
