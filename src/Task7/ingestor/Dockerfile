FROM python:3.10-slim AS init

WORKDIR /app

# Установка системных зависимостей для обработки PDF
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM init AS build
WORKDIR /app

# Создание директорий
RUN mkdir -p documents

# Копирование кода приложения
COPY ./base_ingestor.py ./base_ingestor.py


FROM build AS start

WORKDIR /app

CMD ["python", "-u", "base_ingestor.py"]