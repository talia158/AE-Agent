FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY chunk_markdown.py ./

ENTRYPOINT ["python", "chunk_markdown.py"]
