from fastapi import FastAPI, UploadFile, File
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

@app.get("/")
def read_root():
    return {"message": "AI Video Engine is Running!"}

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    return {"filename": file.filename, "status": "Uploaded Successfully"}