# hand_sign_rulebased_pinky5.py — stabil dengan rule-based
# angka 5 = kelingking aja

import os, cv2, threading, asyncio, collections, hashlib, glob
import mediapipe as mp
import edge_tts
import pygame

VOICE = "id-ID-GadisNeural"   # cowo: "id-ID-ArdiNeural"
RATE  = "-10%"
CACHE_DIR = "voice_cache"

COUNT_HISTORY = 7        # smoothing jumlah jari
MARGIN_RATIO = 0.12      # margin adaptif (12% tinggi tangan)

GESTURE_TEXT = {
    1: "halo kenalin nama saya domi andrey",
    2: "saya talent developer",
    3: "dari caldera indonesia",
    4: "salam kenal dari saya",
    5: "domi andrey",
}

# ---------- TTS cache ----------
def text_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

async def tts_save(text, outpath):
    communicate = edge_tts.Communicate(text, voice=VOICE, rate=RATE)
    with open(outpath, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])

def ensure_cache_sync():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for k, text in GESTURE_TEXT.items():
        key = text_key(text)
        target = os.path.join(CACHE_DIR, f"{k}_{key}.mp3")
        # hapus cache lama utk index k
        for old in glob.glob(os.path.join(CACHE_DIR, f"{k}_*.mp3")):
            if old != target:
                try: os.remove(old)
                except: pass
        if not os.path.exists(target) or os.path.getsize(target) == 0:
            asyncio.run(tts_save(text, target))

def mp3_path_for_index(idx: int) -> str:
    text = GESTURE_TEXT[idx]
    key = text_key(text)
    return os.path.join(CACHE_DIR, f"{idx}_{key}.mp3")

# ---------- Audio ----------
_play_lock = threading.Lock()
def play_mp3(path):
    with _play_lock:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        try:
            pygame.mixer.music.stop()
        except:
            pass
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
TIP_IDS  = [4, 8, 12, 16, 20]

def fingers_up(landmarks, handed_label):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_y, max_y = min(ys), max(ys)
    hand_h = max(1e-6, max_y - min_y)
    margin = MARGIN_RATIO * hand_h

    up = [False]*5
    # Thumb
    tip_x = landmarks[TIP_IDS[0]].x
    ip_x  = landmarks[TIP_IDS[0]-1].x
    if handed_label == "Right":
        up[0] = (tip_x < ip_x - margin)
    else:
        up[0] = (tip_x > ip_x + margin)

    # 4 jari lain
    for idx, tip in enumerate(TIP_IDS[1:], start=1):
        tip_y = landmarks[tip].y
        pip_y = landmarks[tip-2].y
        up[idx] = (tip_y < (pip_y - margin))

    return up

def classify_gesture(fingers):
    """
    fingers = [thumb,index,middle,ring,pinky]
    return angka 1..5 sesuai pola, 0 kalau nggak cocok
    """
    if fingers == [False, True, False, False, False]:
        return 1
    if fingers == [False, True, True, False, False]:
        return 2
    if fingers == [False, True, True, True, False]:
        return 3
    if fingers == [False, True, True, True, True]:
        return 4
    # 5 = cuma kelingking naik
    if fingers == [False, False, False, False, True]:
        return 5
    return 0

# ---------- Main ----------
def main():
    ensure_cache_sync()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam tidak kebuka.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    recent_counts = collections.deque(maxlen=COUNT_HISTORY)
    prev_count = 0

    with mp_hands.Hands(max_num_hands=1, model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            count = 0
            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    fingers = fingers_up(lm.landmark, handed.classification[0].label)
                    count = classify_gesture(fingers)

            # smoothing (majority)
            recent_counts.append(count)
            if recent_counts:
                stable_count = max(set(recent_counts), key=recent_counts.count)
            else:
                stable_count = 0

            # overlay
            if stable_count > 0:
                cv2.putText(frame, f"{stable_count} - {GESTURE_TEXT.get(stable_count,'')}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, "...",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # trigger sekali saat transisi 0 -> N
            if prev_count == 0 and stable_count in GESTURE_TEXT and stable_count != 0:
                mp3 = mp3_path_for_index(stable_count)
                if os.path.exists(mp3) and os.path.getsize(mp3) > 0:
                    threading.Thread(target=play_mp3, args=(mp3,), daemon=True).start()

            prev_count = stable_count

            cv2.imshow("Gesture Voice (q keluar)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
