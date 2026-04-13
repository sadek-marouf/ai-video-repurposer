import sys
import os
import json

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
@app.get("/debug-analysis/{video_name}", tags=["Debug"])
async def debug_analysis(video_name: str):
    base_path = f"processed_data/{video_name}"
    scoring_path = os.path.join(base_path, "scoring.json")
    transcript_path = os.path.join(base_path, "transcript.json")

    if not os.path.exists(scoring_path):
        return {"error": "لم تنتهِ المعالجة بعد أو الملف غير موجود"}

    with open(scoring_path, 'r') as f:
        scores = json.load(f)
    
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)

    return {
        "summary": {
            "total_seconds": len(scores),
            "top_10_moments": scores[:10], # أفضل 10 لحظات رشحها النظام
        },
        "sample_transcript": transcript[:3] # عينة من أول 3 جمل تم فهمها
    }
@app.get("/check-status/{video_name}", tags=["Processing"])
async def check_status(video_name: str):
    # مسار ملف الـ JSON الذي ننتجه في processor.py
    status_path = f"processed_data/{video_name}/transcript.json"
    if os.path.exists(status_path):
        return {"status": "Completed", "data_file": status_path}
    return {"status": "Processing or Not Found"}
