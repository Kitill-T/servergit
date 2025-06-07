FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /workspace

# Копируем зависимости отдельно для кэширования
COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .