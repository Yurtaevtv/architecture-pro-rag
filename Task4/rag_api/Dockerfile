FROM python:3.10-slim AS init

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

FROM init AS build
WORKDIR /app
# Копирование кода
COPY . .

# Создание папок для моделей и логов
RUN mkdir -p /app/models /app/logs /app/faiss_index

FROM build AS start
WORKDIR /app
# Порты
EXPOSE 8001

# Запуск
CMD ["python", "app.py"]