FROM python:3.9-slim

# تثبيت أدوات النظام و FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# الخطوة الأهم: تثبيت Torch CPU بشكل منفصل أولاً لضمان الحجم الصغير
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# الآن تثبيت باقي المكتبات منrequirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي كود المشروع
COPY . .

# إنشاء المجلدات الضرورية
RUN mkdir -p uploads outputs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
