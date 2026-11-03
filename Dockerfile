FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_FORWARDED_ALLOW_IPS=*

WORKDIR /app

# PyMuPDF and numpy ship manylinux wheels, so no compiler toolchain is needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# Coolify/Railway health checks; python:slim has no curl, so use urllib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import os, urllib.request; p = os.getenv('PORT') or '8000'; p = p if p.isdigit() and int(p) > 0 else '8000'; urllib.request.urlopen('http://127.0.0.1:' + p + '/health', timeout=4)"

# One worker only: pairing codes and websocket rooms live in process memory,
# and SQLite is a single-writer file.
# Railway (and most PaaS) inject PORT, so the shell form must expand it; 8000 is the local default.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers"]
