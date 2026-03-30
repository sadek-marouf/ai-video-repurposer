FROM python:3.9-slim

# تثبيت FFmpeg وأدوات النظام
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# تثبيت مكتبات بايثون
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كود التطبيق
COPY . .

# إنشاء مجلدات العمل
RUN mkdir -p uploads outputs

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]