FROM python:3.9-slim

# تثبيت أدوات النظام و FFmpeg بشكل مؤكد
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# تثبيت Torch CPU لتوفير المساحة
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# تثبيت المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كود المشروع بالكامل
COPY . .

# إنشاء المجلدات وضمان الصلاحيات
RUN mkdir -p uploads processed_data && chmod -R 777 uploads processed_data

EXPOSE 8000

# تصحيح مسار التشغيل (تأكد أن main.py موجود داخل مجلد اسمه app أو في الجذور)
# إذا كان main.py في المجلد الرئيسي مباشرة، استخدم: main:app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
