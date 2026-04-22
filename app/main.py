import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from processor import VideoProcessor

app = FastAPI()

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed_data"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=PROCESSED_DIR), name="outputs")

# 📤 رفع فيديو
@app.post("/upload")
async def upload(file: UploadFile = File(...), bg: BackgroundTasks = None):
    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    processor = VideoProcessor(path)

    if bg:
        bg.add_task(processor.process)

    return {
        "status": "processing",
        "video": processor.base_name
    }

# 🔍 الحالة
@app.get("/status/{name}")
def status(name: str):
    p = f"processed_data/{name}/segments.json"

    if os.path.exists(p):
        return {
            "status": "done",
            "download": f"/download/{name}/1"
        }

    return {"status": "processing"}

# 📥 تحميل
@app.get("/download/{name}/{num}")
def download(name: str, num: int):
    path = f"processed_data/{name}/reels/reel_{num}.mp4"

    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4")

    return {"error": "not ready"}
