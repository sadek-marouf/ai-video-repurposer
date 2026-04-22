import os
import subprocess
import cv2
import whisper
import librosa
import numpy as np
import json
import gc

class VideoProcessor:
    def __init__(self, video_path, output_dir="processed_data"):
        self.video_path = os.path.abspath(video_path)
        self.base_name = os.path.basename(video_path).split('.')[0]
        self.output_path = os.path.abspath(os.path.join(output_dir, self.base_name))
        self._setup_dirs()

    def _setup_dirs(self):
        os.makedirs(os.path.join(self.output_path, "audio"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "reels"), exist_ok=True)

    # 🎧 استخراج الصوت
    def extract_audio(self):
        audio_out = os.path.join(self.output_path, "audio", "voice.wav")
        subprocess.run([
            'ffmpeg', '-i', self.video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            audio_out, '-y'
        ], capture_output=True)
        return audio_out

    # 🧠 Whisper
    def transcribe(self, audio_path):
        model = whisper.load_model("tiny", device="cpu")
        result = model.transcribe(audio_path, verbose=False)
        segments = result["segments"]

        del model
        gc.collect()

        return segments

    # 🔊 طاقة الصوت لكل ثانية
    def audio_energy(self, audio_path):
        y, sr = librosa.load(audio_path)
        rms = librosa.feature.rms(y=y)[0]
        duration = int(librosa.get_duration(y=y, sr=sr))
        energy = np.array_split(rms, duration) if duration > 0 else [rms]
        return [float(np.max(e)) if len(e) > 0 else 0 for e in energy]

    # 👁️ تحليل بصري خفيف
    def visual_analysis(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        visual_scores = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % int(fps) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                visual_scores.append(1.0 if len(faces) > 0 else 0.0)

            frame_idx += 1

        cap.release()
        return visual_scores

    # 🧠 حساب score لكل segment
    def score_segments(self, segments, audio_scores, visual_scores):
        results = []

        max_audio = max(audio_scores) if audio_scores else 1.0

        for seg in segments:
            start = int(seg["start"])
            end = int(seg["end"])
            text = seg["text"]

            if end <= start:
                continue

            duration = end - start
            if duration < 2:  # تجاهل المقاطع القصيرة جدًا
                continue

            # صوت
            a = np.mean(audio_scores[start:end]) if end <= len(audio_scores) else 0
            a_norm = a / (max_audio + 1e-6)

            # بصري
            v = np.mean(visual_scores[start:end]) if end <= len(visual_scores) else 0

            # نص
            word_count = len(text.split())
            text_score = min(word_count / 10, 1.0)

            # سكورنغ نهائي
            score = (a_norm * 0.4) + (v * 0.3) + (text_score * 0.3)

            results.append({
                "start": start,
                "end": end,
                "score": round(score, 4),
                "text": text.strip()
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:3]  # أفضل 3 مقاطع

    # 📝 subtitles
    def generate_srt(self, segments, start_limit, duration):
        srt = ""
        counter = 1
        end_limit = start_limit + duration

        def format_time(t):
            hrs = int(t // 3600)
            mins = int((t % 3600) // 60)
            secs = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

        for seg in segments:
            if seg['start'] >= start_limit and seg['end'] <= end_limit:
                s = seg['start'] - start_limit
                e = seg['end'] - start_limit

                srt += f"{counter}\n"
                srt += f"{format_time(s)} --> {format_time(e)}\n"
                srt += f"{seg['text']}\n\n"
                counter += 1

        return srt

    # 🎬 إنشاء الريلز
    def generate_reels(self, best_segments, segments):
        outputs = []

        for i, seg in enumerate(best_segments):
            start = max(0, seg["start"] - 2)
            duration = min(15, seg["end"] - seg["start"] + 4)

            srt_text = self.generate_srt(segments, start, duration)
            srt_path = os.path.join(self.output_path, "reels", f"s{i}.srt")

            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_text)

            out = os.path.join(self.output_path, "reels", f"reel_{i+1}.mp4")

            vf = (
                "scale=720:1280:force_original_aspect_ratio=increase,"
                "crop=720:1280,"
                f"subtitles='{srt_path}':force_style='FontSize=16'"
            )

            subprocess.run([
                "ffmpeg",
                "-ss", str(start),
                "-t", str(duration),
                "-i", self.video_path,
                "-vf", vf,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-c:a", "aac",
                out,
                "-y"
            ], capture_output=True)

            if os.path.exists(out):
                outputs.append(out)

        return outputs

    # 🚀 pipeline
    def process(self):
        audio = self.extract_audio()
        segments = self.transcribe(audio)

        audio_scores = self.audio_energy(audio)
        visual_scores = self.visual_analysis()

        best = self.score_segments(segments, audio_scores, visual_scores)

        with open(os.path.join(self.output_path, "segments.json"), "w") as f:
            json.dump(best, f, indent=4)

        reels = self.generate_reels(best, segments)

        return reels
