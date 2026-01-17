FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY convert_to_md.py ./

ENTRYPOINT ["python", "convert_to_md.py"]
