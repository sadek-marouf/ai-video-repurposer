import os
import subprocess
import cv2
import whisper
import librosa
import numpy as np
import json

class VideoProcessor:
    def __init__(self, video_path, output_dir="processed_data"):
        self.video_path = video_path
        self.base_name = os.path.basename(video_path).split('.')[0]
        self.output_path = os.path.join(output_dir, self.base_name)
        self._setup_dirs()
        
        # تحميل موديل Whisper (سأستخدم النسخة base لسرعة السيرفر)
        self.model = whisper.load_model("base")

    def _setup_dirs(self):
        """تهيئة المجلدات المطلوبة للعمل"""
        os.makedirs(os.path.join(self.output_path, "audio"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "reels"), exist_ok=True)

    def step_2_extract_audio(self):
        """استخراج الصوت بجودة احترافية للتحليل"""
        audio_out = os.path.join(self.output_path, "audio", "voice.wav")
        command = [
            'ffmpeg', '-i', self.video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            audio_out, '-y'
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return audio_out

    def step_3_whisper_analysis(self, audio_path):
        """تحويل الكلام لنص مع التوقيت (Word-level timestamps)"""
        print(f"--- Processing Whisper for: {self.base_name} ---")
        result = self.model.transcribe(audio_path, verbose=False)
        
        # تخزين النص مع التوقيت في ملف JSON للرجوع له لاحقاً
        transcript_path = os.path.join(self.output_path, "transcript.json")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result['segments'], f, ensure_ascii=False, indent=4)
            
        return result['segments']

    def step_4_audio_energy(self, audio_path):
        """تحليل طاقة الصوت (Energy) لتحديد اللحظات الحماسية"""
        y, sr = librosa.load(audio_path)
        
        # حساب الـ RMS (الطاقة الصوتية)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr)
        
        # تحويلها لـ Dictionary يسهل البحث فيه (ثانية: درجة الطاقة)
        energy_map = {round(t, 2): float(e) for t, e in zip(times, rms)}
        return energy_map

    def process_pipeline(self):
        """تشغيل المرحلة الأولى من الخطة"""
        # 1. استخراج الصوت
        audio_file = self.step_2_extract_audio()
        
        # 2. تحليل النص (Whisper)
        segments = self.step_3_whisper_analysis(audio_file)
        
        # 3. تحليل الطاقة (Librosa)
        energy_data = self.step_4_audio_energy(audio_file)
        
        return {
            "status": "Ready for scoring",
            "segments_count": len(segments),
            "audio_analysis": "Complete"
        }

# لتجربة الكود يدوياً
if __name__ == "__main__":
    # تأكد من وجود فيديو باسم test.mp4 بجانب الملف
    proc = VideoProcessor("test.mp4")
    print(proc.process_pipeline())
