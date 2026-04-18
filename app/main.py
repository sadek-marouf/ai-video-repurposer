import sys
import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# إضافة المسار الحالي لمسارات بايثون
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from processor import VideoProcessor

app = FastAPI(title="Video to Reels Processor")

# مجلدات النظام
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# تفعيل الوصول للملفات عبر المتصفح (مهم جداً لمشاهدة الريلز)
app.mount("/outputs", StaticFiles(directory=PROCESSED_DIR), name="outputs")

@app.post("/upload-and-process/", tags=["Processing"])
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    processor = VideoProcessor(file_path)
    
    # تشغيل المعالجة في الخلفية
    background_tasks.add_task(processor.process_pipeline)
    
    return {
        "message": "بدأت عملية المعالجة في الخلفية",
        "file_name": file.filename,
        "output_folder": processor.output_path,
        "debug_url": f"/debug-analysis/{processor.base_name}"
    }

@app.get("/debug-analysis/{video_name}", tags=["Debug"])
async def debug_analysis(video_name: str):
    base_path = f"processed_data/{video_name}"
    scoring_path = os.path.join(base_path, "scoring.json")
    transcript_path = os.path.join(base_path, "transcript.json")

    if not os.path.exists(scoring_path):
        return {"error": "المعالجة مستمرة أو الملف غير موجود"}

    with open(scoring_path, 'r') as f:
        scores = json.load(f)
    
    with open(transcript_path, 'r') as f:
        transcript = json.load(f)

    return {
        "summary": {
            "total_seconds": len(scores),
            "top_10_moments": scores[:10],
        },
        "sample_transcript": transcript[:3]
    }

@app.get("/check-status/{video_name}", tags=["Processing"])
async def check_status(video_name: str):
    status_path = f"processed_data/{video_name}/scoring.json"
    if os.path.exists(status_path):
        return {
            "status": "Completed", 
            "scoring_file": status_path,
            "reels_dir": f"/outputs/{video_name}/reels"
        }
    return {"status": "Processing..."}
@app.get("/download-reel/{video_name}/{reel_number}", tags=["Download"])
async def download_reel(video_name: str, reel_number: int):
    # مسار الريل المطلوب
    reel_path = f"processed_data/{video_name}/reels/reel_{reel_number}.mp4"
    
    if os.path.exists(reel_path):
        return FileResponse(
            path=reel_path, 
            filename=f"{video_name}_reel_{reel_number}.mp4", 
            media_type='video/mp4'
        )
    return {"error": "الملف غير موجود، تأكد من انتهاء المعالجة"}
