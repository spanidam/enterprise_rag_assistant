FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (for PyMuPDF, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "127.0.0.1", "--port", "8000"]