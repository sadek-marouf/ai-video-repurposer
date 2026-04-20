import sys
import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# إضافة المسار الحالي
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from processor import VideoProcessor

app = FastAPI(title="Video to Reels Processor")

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=PROCESSED_DIR), name="outputs")

@app.post("/upload-and-process/", tags=["Processing"])
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    processor = VideoProcessor(file_path)
    background_tasks.add_task(processor.process_pipeline)
    
    return {
        "message": "بدأت عملية المعالجة في الخلفية",
        "file_name": file.filename,
        "video_id": processor.base_name
    }

@app.get("/check-status/{video_name}", tags=["Processing"])
async def check_status(video_name: str):
    status_path = f"processed_data/{video_name}/scoring.json"
    if os.path.exists(status_path):
        return {
            "status": "Completed", 
            "download_url": f"/download-reel/{video_name}/1"
        }
    return {"status": "Processing... Please wait"}

@app.get("/download-reel/{video_name}/{reel_number}", tags=["Download"])
async def download_reel(video_name: str, reel_number: int):
    reel_path = os.path.abspath(f"processed_data/{video_name}/reels/reel_{reel_number}.mp4")
    
    if os.path.exists(reel_path):
        return FileResponse(
            path=reel_path, 
            filename=f"{video_name}_reel_{reel_number}.mp4", 
            media_type='video/mp4',
            headers={"Content-Disposition": f"attachment; filename={video_name}_reel.mp4"}
        )
    return {"error": "الملف غير موجود بعد"}

@app.get("/debug-analysis/{video_name}", tags=["Debug"])
async def debug_analysis(video_name: str):
    base_path = f"processed_data/{video_name}"
    scoring_path = os.path.join(base_path, "scoring.json")
    if not os.path.exists(scoring_path):
        return {"error": "لا توجد بيانات بعد"}
    with open(scoring_path, 'r') as f:
        scores = json.load(f)
    return {"top_moments": scores[:5]}
