FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY semantic_encoder.py ./

ENTRYPOINT ["python", "semantic_encoder.py"]
