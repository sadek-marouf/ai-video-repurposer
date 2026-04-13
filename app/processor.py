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
        
        # تحميل موديل Whisper (النسخة base متوازنة بين السرعة والدقة)
        self.model = whisper.load_model("base")

    def _setup_dirs(self):
        """تهيئة المجلدات المطلوبة للعمل لكل فيديو"""
        os.makedirs(os.path.join(self.output_path, "audio"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "reels"), exist_ok=True)

    def step_2_extract_audio(self):
        """الخطوة 2: استخراج الصوت بصيغة WAV للتحليل"""
        audio_out = os.path.join(self.output_path, "audio", "voice.wav")
        command = [
            'ffmpeg', '-i', self.video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            audio_out, '-y'
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return audio_out

    def step_3_whisper_analysis(self, audio_path):
        """الخطوة 3: استخراج النصوص والكلمات المفتاحية"""
        print(f"--- Running Whisper Analysis ---")
        result = self.model.transcribe(audio_path, verbose=False)
        transcript_path = os.path.join(self.output_path, "transcript.json")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result['segments'], f, ensure_ascii=False, indent=4)
        return result['segments']

    def step_4_audio_energy(self, audio_path):
        """الخطوة 4: تحليل قوة الصوت (الحماس) لكل ثانية"""
        y, sr = librosa.load(audio_path)
        rms = librosa.feature.rms(y=y)[0]
        # تقسيم الطاقة إلى ثوانٍ فعلية
        energy_per_second = np.array_split(rms, int(librosa.get_duration(y=y, sr=sr)))
        return [float(np.max(e)) if len(e) > 0 else 0 for e in energy_per_second]

    def step_5_visual_analysis(self):
        """الخطوة 5: التحليل البصري (كشف الوجوه)"""
        print(f"--- Running Visual Analysis (Face Detection) ---")
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        visual_scores = []
        
        # استخدام Haar Cascade البسيط (سريع ومناسب للسيرفر)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # نحلل إطار واحد لكل ثانية لتوفير الوقت
            if count % int(fps) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                # إذا وجد وجه يأخذ 1 وإلا 0
                visual_scores.append(1.0 if len(faces) > 0 else 0.0)
            count += 1
            
        cap.release()
        return visual_scores

    def step_6_calculate_scores(self, segments, audio_scores, visual_scores):
        final_ranking = []
        duration = min(len(audio_scores), len(visual_scores))
        
        # تحويل درجات الصوت لقيم بين 0 و 1 (Normalization)
        max_audio = max(audio_scores) if max_audio > 0 else 1.0
        min_audio = min(audio_scores)
        
        for sec in range(duration):
            # 1. سكور الصوت: نجعل أعلى صوت في المقطع هو 1.0
            a_norm = (audio_scores[sec] - min_audio) / (max_audio - min_audio + 1e-6)
            
            # 2. سكور الوجه: موجود (1) أو غير موجود (0)
            v_score = visual_scores[sec]
            
            # 3. سكور النص: هل توجد جملة تبدأ أو تنتهي في هذه الثانية؟
            t_score = 0.0
            for seg in segments:
                if seg['start'] <= sec <= seg['end']:
                    t_score = 1.0 # هذه الثانية تحتوي على كلام
                    break

            # المعادلة الجديدة (وزن أكبر للصوت والوجه في الأنميشن)
            # 40% صوت، 40% وجه، 20% وجود كلام
            combined_score = (a_norm * 0.4) + (v_score * 0.4) + (t_score * 0.2)
            
            final_ranking.append({
                "second": sec,
                "score": round(combined_score, 4)
            })
            
        final_ranking.sort(key=lambda x: x['score'], reverse=True)
        return final_ranking

    def process_pipeline(self):
        """تنفيذ العملية الكاملة المخطط لها"""
        audio_file = self.step_2_extract_audio()
        segments = self.step_3_whisper_analysis(audio_file)
        audio_scores = self.step_4_audio_energy(audio_file)
        visual_scores = self.step_5_visual_analysis()
        
        # حساب النقاط النهائية
        top_moments = self.step_6_calculate_scores(segments, audio_scores, visual_scores)
        
        # حفظ الترتيب النهائي
        with open(os.path.join(self.output_path, "scoring.json"), 'w') as f:
            json.dump(top_moments, f, indent=4)
            
        return {
            "status": "Success",
            "top_moments": top_moments[:5], # عرض أفضل 5 ثوانٍ في النتيجة
            "output_dir": self.output_path
        }

if __name__ == "__main__":
    proc = VideoProcessor("test.mp4")
    print(proc.process_pipeline())
