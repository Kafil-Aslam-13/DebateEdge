FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt setup.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN --mount=type=secret,id=guardrails_api_key \
    guardrails configure \
      --token "$(cat /run/secrets/guardrails_api_key)" \
      --enable-metrics false \
      --disable-remote-inferencing && \
    guardrails hub install hub://guardrails/detect_pii --quiet && \
    guardrails hub install hub://guardrails/toxic_language --quiet && \
    guardrails hub install hub://guardrails/gibberish_text --quiet


COPY src ./src
COPY app/api ./app/api
COPY configs ./configs

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
