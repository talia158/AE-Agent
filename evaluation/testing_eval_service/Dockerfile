FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY testing_eval_service.py ./

ENTRYPOINT ["python", "testing_eval_service.py"]
