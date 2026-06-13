FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir ultralytics opencv-python-headless numpy

CMD ["python", "app.py"]