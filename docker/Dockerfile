FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ARG GUARDRAILS_API_KEY
RUN guardrails configure --token ${GUARDRAILS_API_KEY} --enable-metrics false || guardrails configure --disable-metrics --disable-remote-inferencing
RUN guardrails hub install hub://guardrails/detect_pii --quiet
RUN guardrails hub install hub://guardrails/toxic_language --quiet
RUN guardrails hub install hub://guardrails/gibberish_text --quiet


COPY src     ./src
COPY app/api ./app/api
COPY configs ./configs


EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]