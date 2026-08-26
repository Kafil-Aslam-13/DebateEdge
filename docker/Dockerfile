FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt setup.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY src ./src
COPY app/api ./app/api
COPY configs ./configs

COPY docker/start.sh /code/start.sh
RUN chmod +x /code/start.sh

EXPOSE 8000

CMD ["/code/start.sh"]