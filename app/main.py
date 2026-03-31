from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .processor import VideoProcessor  # استدعاء المحرك الذي كتبناه
import os

app = FastAPI()

# إضافة تصريح الـ CORS لضمان عمل الواجهة والمتصفح
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إنشاء كائن من المعالج
processor = VideoProcessor()

@app.get("/")
def read_root():
    return {"message": "AI Video Engine is Running!"}

# تغيير الـ Endpoint إلى /process/ ليعبر عن المعالجة
@app.post("/process/")
async def process_video(file: UploadFile = File(...)):
    # 1. حفظ الملف مؤقتاً
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # 2. استدعاء Whisper لمعالجة الفيديو (هنا يحدث السحر)
        transcription = await processor.transcribe_video(temp_path)
        
        # 3. مسح الملف المؤقت بعد الانتهاء لتوفير المساحة
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "filename": file.filename, 
            "status": "Success",
            "data": transcription # هنا ستظهر النصوص والتايم كود
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}
