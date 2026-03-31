import sys
import os

# إضافة المسار الحالي لمسارات بايثون
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from processor import VideoProcessor
import shutil
import os

app = FastAPI(title="Video to Reels Processor")

# مجلد مؤقت لحفظ الفيديوهات المرفوعة
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-and-process/", tags=["Processing"])
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 1. حفظ الملف محلياً
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. إنشاء كائن المعالج
    processor = VideoProcessor(file_path)
    
    # 3. تشغيل المعالجة (سنشغلها كـ Background Task لكي لا ينتظر Swagger طويلاً)
    background_tasks.add_task(processor.process_pipeline)
    
    return {
        "message": "بدأت عملية المعالجة في الخلفية",
        "file_name": file.filename,
        "output_folder": processor.output_path
    }

@app.get("/check-status/{video_name}", tags=["Processing"])
async def check_status(video_name: str):
    # مسار ملف الـ JSON الذي ننتجه في processor.py
    status_path = f"processed_data/{video_name}/transcript.json"
    if os.path.exists(status_path):
        return {"status": "Completed", "data_file": status_path}
    return {"status": "Processing or Not Found"}
