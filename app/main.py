import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed_data"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=PROCESSED_DIR), name="outputs")

# 📤 رفع فيديو (بدون معالجة)
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_ext = file.filename.split('.')[-1]

    filename = f"{job_id}.{file_ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "status": "uploaded",
        "job_id": job_id,
        "filename": filename
    }

# 🔍 الحالة
@app.get("/status/{job_id}")
def status(job_id: str):
    base_name = job_id
    p = f"processed_data/{base_name}/segments.json"

    if os.path.exists(p):
        return {
            "status": "done",
            "download": f"/download/{base_name}/1"
        }

    return {"status": "waiting for processing"}

# 📥 تحميل
@app.get("/download/{job_id}/{num}")
def download(job_id: str, num: int):
    path = f"processed_data/{job_id}/reels/reel_{num}.mp4"

    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4")

    return {"error": "not ready"}
@app.get("/get-video/{filename}")
def get_video(filename: str):
    path = os.path.join("uploads", filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4")
    return {"error": "not found"}
