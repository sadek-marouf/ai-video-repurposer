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
        self.model = whisper.load_model("base")

    def _setup_dirs(self):
        os.makedirs(os.path.join(self.output_path, "audio"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "reels"), exist_ok=True)

    def step_2_extract_audio(self):
        audio_out = os.path.join(self.output_path, "audio", "voice.wav")
        command = [
            'ffmpeg', '-i', self.video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            audio_out, '-y'
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return audio_out

    def step_3_whisper_analysis(self, audio_path):
        result = self.model.transcribe(audio_path, verbose=False)
        transcript_path = os.path.join(self.output_path, "transcript.json")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result['segments'], f, ensure_ascii=False, indent=4)
        return result['segments']

    def step_4_audio_energy(self, audio_path):
        y, sr = librosa.load(audio_path)
        rms = librosa.feature.rms(y=y)[0]
        duration = int(librosa.get_duration(y=y, sr=sr))
        energy_per_second = np.array_split(rms, duration) if duration > 0 else [rms]
        return [float(np.max(e)) if len(e) > 0 else 0 for e in energy_per_second]

    def step_5_visual_analysis(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        visual_scores = []
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % int(fps or 30) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                visual_scores.append(1.0 if len(faces) > 0 else 0.0)
            count += 1
        cap.release()
        return visual_scores

    def step_6_calculate_scores(self, segments, audio_scores, visual_scores):
        final_ranking = []
        duration = min(len(audio_scores), len(visual_scores))
        
        # تصحيح حسابات الـ Audio (Normalization)
        m_audio = max(audio_scores) if audio_scores and max(audio_scores) > 0 else 1.0
        min_audio = min(audio_scores) if audio_scores else 0.0
        diff = m_audio - min_audio

        for sec in range(duration):
            a_norm = (audio_scores[sec] - min_audio) / (diff + 1e-6)
            v_score = visual_scores[sec]
            
            t_score = 0.0
            for seg in segments:
                if seg['start'] <= sec <= seg['end']:
                    t_score = 1.0
                    break

            # ميزان الجيمنج/الأنميشن الجديد
            combined_score = (a_norm * 0.4) + (v_score * 0.4) + (t_score * 0.2)
            final_ranking.append({"second": sec, "score": round(combined_score, 4)})
            
        final_ranking.sort(key=lambda x: x['score'], reverse=True)
        return final_ranking
    def step_8_generate_reels(self, top_moments, count=1):
        """الخطوة 8: قص الفيديو وتحويله لأبعاد الريلز (9:16)"""
        print(f"--- Generating Reels ---")
        reels_created = []

        for i in range(min(count, len(top_moments))):
            best_sec = top_moments[i]['second']
            # تحديد وقت البداية (قبل اللحظة بـ 2 ثانية) والمدة (6 ثوانٍ للريل القصير)
            start_time = max(0, best_sec - 2)
            duration = 6 
            
            output_file = os.path.join(self.output_path, "reels", f"reel_{i+1}.mp4")
            
            # أمر FFmpeg: القص + التحويل لعمودي + Scale لضمان الجودة
            command = [
                'ffmpeg', '-ss', str(start_time), '-t', str(duration),
                '-i', self.video_path,
                '-vf', "crop=ih*(9/16):ih,scale=720:1280", 
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                output_file, '-y'
            ]
            
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            reels_created.append(output_file)
        
        return reels_created

    def process_pipeline(self):
        """النسخة الكاملة من المسار"""
        try:
            audio_file = self.step_2_extract_audio()
            segments = self.step_3_whisper_analysis(audio_file)
            audio_scores = self.step_4_audio_energy(audio_file)
            visual_scores = self.step_5_visual_analysis()
            
            top_moments = self.step_6_calculate_scores(segments, audio_scores, visual_scores)
            
            # حفظ السكور
            with open(os.path.join(self.output_path, "scoring.json"), 'w') as f:
                json.dump(top_moments, f, indent=4)
            
            # توليد الريلز فعلياً
            self.step_8_generate_reels(top_moments)
            
            print(f"--- Reels Generated Successfully! ---")
            return True
        except Exception as e:
            print(f"--- Error: {str(e)} ---")
            return False
