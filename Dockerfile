FROM python:3.9-slim

# تثبيت أدوات النظام، FFmpeg، ومكتبات OpenCV المتوافقة مع النسخ الجديدة
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# تثبيت Torch CPU أولاً لتقليل المساحة (Railway لديه ليميت في الحجم أحياناً)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# نسخ وتثبيت المكتبات (تأكد أن opencv-python-headless موجود في requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كود المشروع
COPY . .

# إنشاء مجلدات العمل لضمان عدم وجود أخطاء Permission
RUN mkdir -p uploads processed_data

EXPOSE 8000

# تشغيل السيرفر (تأكد من مسار المديول حسب هيكلية ملفاتك)
CMD ["uvicorn","app.main:app", "--host", "0.0.0.0", "--port", "8000"]
