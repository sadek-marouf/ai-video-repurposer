FROM python:3.9-slim

# 1. تثبيت أدوات النظام، FFmpeg، والخطوط العربية في أمر واحد
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu \
    fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. تثبيت Torch CPU
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. تثبيت المكتبات الأخرى
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. نسخ كود المشروع
COPY . .

# 5. إنشاء المجلدات وتجهيز الصلاحيات
RUN mkdir -p uploads processed_data && chmod -R 777 uploads processed_data

EXPOSE 8000

# 6. تشغيل السيرفر
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
