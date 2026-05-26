import cv2
import numpy as np
import time
import threading
import os
import pygame
from collections import deque
from mutagen.mp3 import MP3
from deepface import DeepFace

AUDIO_FILE = "sayakanlawan.mp3"
MASK_IMAGE = "jokowi.png"

ANGRY_THRESHOLD = 1.0             
ANGRY_SCORE_MIN = 0.20            
ANGRY_BOOST = 1.5                
DISGUST_BOOST = 0.7             
ANALYZE_EVERY_N_FRAMES = 3       
SMOOTHING_WINDOW = 4             
COOLDOWN_AFTER_TRIGGER = 2.0     

is_playing = False
angry_start_time = None
last_trigger_end_time = 0
angry_history = deque(maxlen=SMOOTHING_WINDOW) 

def play_audio_thread(filepath):
    global is_playing, last_trigger_end_time
    try:
        pygame.mixer.music.load(filepath)
        is_playing = True
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"Error audio: {e}")
    finally:
        is_playing = False
        last_trigger_end_time = time.time()

def start_angry_effect():
    threading.Thread(target=play_audio_thread, args=(AUDIO_FILE,), daemon=True).start()

def overlay_image(background, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    bh, bw = background.shape[:2]
    if x < 0:
        overlay_resized = overlay_resized[:, -x:]; w += x; x = 0
    if y < 0:
        overlay_resized = overlay_resized[-y:, :]; h += y; y = 0
    if x + w > bw:
        overlay_resized = overlay_resized[:, :bw - x]; w = bw - x
    if y + h > bh:
        overlay_resized = overlay_resized[:bh - y, :]; h = bh - y
    if w <= 0 or h <= 0:
        return background

    if overlay_resized.shape[2] == 4:
        alpha = overlay_resized[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha * overlay_resized[:, :, c] +
                (1 - alpha) * background[y:y+h, x:x+w, c]
            )
    else:
        background[y:y+h, x:x+w] = overlay_resized
    return background

def compute_anger_score(emotions):
    angry = emotions.get("angry", 0) / 100.0
    disgust = emotions.get("disgust", 0) / 100.0
    fear = emotions.get("fear", 0) / 100.0
    happy = emotions.get("happy", 0) / 100.0
    surprise = emotions.get("surprise", 0) / 100.0
    sad = emotions.get("sad", 0) / 100.0

    score = (angry * ANGRY_BOOST) + (disgust * DISGUST_BOOST) + (fear * 0.2)
    score -= (happy * 1.5) + (surprise * 0.5)
    score += sad * 0.15
    return max(0.0, score)

def main():
    global angry_start_time, is_playing

    if not os.path.exists(AUDIO_FILE):
        print(f"[ERROR] {AUDIO_FILE} tidak ditemukan!"); return
    if not os.path.exists(MASK_IMAGE):
        print(f"[ERROR] {MASK_IMAGE} tidak ditemukan!"); return

    pygame.mixer.init()
    try:
        audio_duration = MP3(AUDIO_FILE).info.length
        print(f"[INFO] Durasi audio: {audio_duration:.2f}s")
    except: pass

    mask_img = cv2.imread(MASK_IMAGE, cv2.IMREAD_UNCHANGED)
    if mask_img is None:
        print(f"[ERROR] Gagal baca {MASK_IMAGE}"); return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam tidak bisa dibuka!"); return

    print("[INFO] Berjalan. Tekan 'q' keluar, 'd' debug mode (lihat semua emosi).")

    frame_count = 0
    last_face = None
    last_emotions = {}     
    current_anger = 0.0
    debug_mode = True       

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            last_face = faces[0]

        frame_count += 1
        in_cooldown = (time.time() - last_trigger_end_time) < COOLDOWN_AFTER_TRIGGER

        if (frame_count % ANALYZE_EVERY_N_FRAMES == 0
                and last_face is not None
                and not is_playing
                and not in_cooldown):
            try:
                x, y, w, h = last_face
                pad = 30
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                face_roi = frame[y1:y2, x1:x2]

                result = DeepFace.analyze(
                    face_roi,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",  
                    silent=True,
                )
                if isinstance(result, list): result = result[0]

                emotions = result["emotion"]
                last_emotions = emotions

                anger = compute_anger_score(emotions)
                angry_history.append(anger)

                current_anger = sum(angry_history) / len(angry_history)

                if current_anger >= ANGRY_SCORE_MIN:
                    if angry_start_time is None:
                        angry_start_time = time.time()
                        print(f"[INFO] Marah mulai terdeteksi (score: {current_anger:.2f})")
                    elif time.time() - angry_start_time >= ANGRY_THRESHOLD:
                        print(f"[INFO] >>> TRIGGER! score: {current_anger:.2f}")
                        start_angry_effect()
                        angry_start_time = None
                        angry_history.clear()
                else:
                    if angry_start_time is not None and current_anger < ANGRY_SCORE_MIN * 0.6:
                        angry_start_time = None

            except Exception as e:
                pass

        if is_playing and last_face is not None:
            x, y, w, h = last_face
            scale = 1.3
            new_w = int(w * scale); new_h = int(h * scale)
            new_x = x - (new_w - w) // 2; new_y = y - (new_h - h) // 2
            frame = overlay_image(frame, mask_img, new_x, new_y, new_w, new_h)
        else:
            if last_face is not None:
                x, y, w, h = last_face 
                if current_anger >= ANGRY_SCORE_MIN:
                    color = (0, 0, 255)      
                elif current_anger >= ANGRY_SCORE_MIN * 0.6:
                    color = (0, 165, 255)  
                else:
                    color = (0, 255, 0)  
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"Anger: {current_anger:.2f}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

        if is_playing:
            status, color = "MARAH AKTIF!", (0, 0, 255)
        elif in_cooldown:
            status, color = "Cooldown...", (128, 128, 128)
        elif angry_start_time is not None:
            elapsed = time.time() - angry_start_time
            status = f"Mulai marah... {elapsed:.1f}s / {ANGRY_THRESHOLD}s"
            color = (0, 165, 255)
        else:
            status, color = "Memantau...", (255, 255, 255)

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if debug_mode and last_emotions:
            y_off = 60
            for emo, val in sorted(last_emotions.items(), key=lambda x: -x[1]):
                txt = f"{emo:10s}: {val:5.1f}%"
                cv2.putText(frame, txt, (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)
                y_off += 20

        cv2.imshow("Angry Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('d'): debug_mode = not debug_mode

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()