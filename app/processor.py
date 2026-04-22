import os
import subprocess
import cv2
import whisper
import librosa
import numpy as np
import json
import gc # مكتبة تنظيف الذاكرة

class VideoProcessor:
    def __init__(self, video_path, output_dir="processed_data"):
        self.video_path = os.path.abspath(video_path)
        self.base_name = os.path.basename(video_path).split('.')[0]
        self.output_path = os.path.abspath(os.path.join(output_dir, self.base_name))
        self._setup_dirs()
        # حذفنا تحميل الموديل من هنا لتوفير الذاكرة

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
        subprocess.run(command, capture_output=True)
        return audio_out

    def step_3_whisper_analysis(self, audio_path):
        print("--- 🧠 Loading Whisper (tiny) into Memory ---")
        # استخدام tiny لتوفير حوالي 400MB من الرام
        model = whisper.load_model("tiny", device="cpu")
        result = model.transcribe(audio_path, verbose=False)
        
        transcript_path = os.path.join(self.output_path, "transcript.json")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result['segments'], f, ensure_ascii=False, indent=4)
        
        # حذف الموديل فوراً لتفريغ الرام
        del model
        gc.collect() 
        print("--- 🗑️ Memory Cleared ---")
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
            combined_score = (a_norm * 0.4) + (v_score * 0.4) + (t_score * 0.2)
            final_ranking.append({"second": sec, "score": round(combined_score, 4)})
            
        final_ranking.sort(key=lambda x: x['score'], reverse=True)
        return final_ranking

    def _generate_srt(self, segments, start_limit, duration):
        """تحويل قطع نص Whisper إلى ملف SRT متوافق مع وقت الريل"""
        srt_content = ""
        end_limit = start_limit + duration
        
        counter = 1
        for seg in segments:
            # التحقق مما إذا كان النص يقع ضمن توقيت الريل
            if seg['start'] >= start_limit and seg['end'] <= end_limit:
                s_time = seg['start'] - start_limit
                e_time = seg['end'] - start_limit
                
                # تنسيق الوقت لصيغة SRT: HH:MM:SS,ms
                def format_time(seconds):
                    hrs = int(seconds // 3600)
                    mins = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    milli = int((seconds - int(seconds)) * 1000)
                    return f"{hrs:02}:{mins:02}:{secs:02},{milli:03}"

                srt_content += f"{counter}\n"
                srt_content += f"{format_time(s_time)} --> {format_time(e_time)}\n"
                srt_content += f"{seg['text'].strip()}\n\n"
                counter += 1
        return srt_content

    def step_8_generate_reels(self, top_moments, segments, count=1):
        print(f"--- 🎬 Generating Pro Reels with Subtitles ---")
        reels_created = []

        for i in range(min(count, len(top_moments))):
            best_sec = top_moments[i]['second']
            start_time = max(0, best_sec - 3)
            duration = 10 
            
            # 1. إنشاء ملف SRT مؤقت لهذا الريل
            srt_text = self._generate_srt(segments, start_time, duration)
            srt_path = os.path.join(self.output_path, "reels", f"sub_{i+1}.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_text)

            output_file = os.path.join(self.output_path, "reels", f"reel_{i+1}.mp4")
            
            # 2. إعداد فلاتر FFmpeg: القص + الترجمة
            # ملاحظة: يجب أن يكون مسار ملف SRT في فلتر subtitles مهيأ بشكل خاص للـ Linux
            escaped_srt_path = srt_path.replace(":", "\\:").replace("\\", "/")
            
            # تنسيق الخط: Fontsize=18, PrimaryColour (أبيض مع حدود سوداء)
            vf_filters = (
                f"scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,"
                f"subtitles='{escaped_srt_path}':force_style='FontName=DejaVu Sans,Alignment=2,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=1'"
            )
            
            command = [
                'ffmpeg', '-y', 
                '-ss', str(start_time), 
                '-t', str(duration),
                '-i', self.video_path,
                '-vf', vf_filters,
                '-c:v', 'libx264', '-crf', '20', 
                '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '128k',
                output_file
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
                print(f"✅ Pro Reel {i+1} Created with Subtitles!")
                reels_created.append(output_file)
            else:
                print(f"❌ Error: {result.stderr}")
        
        return reels_created

    def process_pipeline(self):
        try:
            # 1. استخراج الصوت
            audio_file = self.step_2_extract_audio()
            
            # 2. تحليل النصوص (Whisper) والحصول على السجمنتس
            segments = self.step_3_whisper_analysis(audio_file)
            
            # 3. تحليل الطاقة الصوتية والبصرية
            audio_scores = self.step_4_audio_energy(audio_file)
            visual_scores = self.step_5_visual_analysis()
            
            # 4. حساب أفضل اللحظات
            top_moments = self.step_6_calculate_scores(segments, audio_scores, visual_scores)
            
            # 5. حفظ النتائج في ملف JSON للرجوع إليها
            with open(os.path.join(self.output_path, "scoring.json"), 'w') as f:
                json.dump(top_moments, f, indent=4)
            
            # 6. تمرير السجمنتس لعملية توليد الريلز (هنا كان الخطأ)
            self.step_8_generate_reels(top_moments, segments)
            
            print("--- ✨ Pipeline Finished Successfully! ---")
            return True
        except Exception as e:
            print(f"--- ❌ Pipeline Failed: {str(e)} ---")
            return False
