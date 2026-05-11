import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import socket
from pathlib import Path
from flask import Flask, Response, render_template, jsonify, request
import sounddevice as sd

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "pose_landmarker_full.task"
MODELS_DIR = BASE_DIR / "models"

# Shared state

class State:
    def __init__(self):
        self.lock        = threading.Lock()
        self.exercise    = None
        self.rep_count   = 0
        self.last_result = None
        self.in_rep      = False
        self.frame_jpg   = None

state = State()

# Config
REP_CONFIG = {
    "squat": {"signal": "knee",  "mode": "valley", "start_thresh": 140, "end_thresh": 155},
    "ohp":   {"signal": "elbow", "mode": "peak",   "start_thresh": 120, "end_thresh": 100},
    "dlift": {"signal": "hip",   "mode": "valley", "start_thresh": 150, "end_thresh": 165},
}

FEATURES = {
    "squat": [
        "knee_angle_min","knee_angle_max","knee_angle_mean","knee_angle_std","knee_angle_range",
        "hip_angle_min","hip_angle_max","hip_angle_mean","hip_angle_std","hip_angle_range",
        "angle_torso_min","angle_torso_max","angle_torso_mean","angle_torso_std","angle_torso_range",
    ],
    "ohp": [
        "elbow_angle_min","elbow_angle_max","elbow_angle_mean","elbow_angle_std","elbow_angle_range",
        "shoulder_angle_min","shoulder_angle_max","shoulder_angle_mean","shoulder_angle_std","shoulder_angle_range",
    ],
    "dlift": [
        "hip_angle_min","hip_angle_max","hip_angle_mean","hip_angle_std","hip_angle_range",
        "knee_angle_min","knee_angle_max","knee_angle_mean","knee_angle_std","knee_angle_range",
        "angle_torso_min","angle_torso_max","angle_torso_mean","angle_torso_std","angle_torso_range",
    ],
}

CONNECTIONS = [
    ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
    ("right_shoulder","right_hip"),  ("right_hip","right_knee"),
    ("right_knee","right_ankle"),    ("left_shoulder","right_shoulder"),
    ("left_hip","right_hip"),        ("left_shoulder","left_elbow"),
    ("left_hip","left_knee"),
]

LANDMARK_IDX = {
    "left_shoulder":11,"right_shoulder":12,
    "left_elbow":13,  "right_elbow":14,
    "left_wrist":15,  "right_wrist":16,
    "left_hip":23,    "right_hip":24,
    "left_knee":25,   "right_knee":26,
    "left_ankle":27,  "right_ankle":28,
}

VIS_THRESHOLD = 0.5
BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode


# When I want to have a phone as a camera
def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def play_ping(freq=1050, duration=0.18, volume=0.6, sample_rate=44100):
    t        = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    envelope = np.exp(-t * 18)
    wave     = (np.sin(2 * np.pi * freq * t) * envelope * volume).astype(np.float32)
    sd.play(wave, samplerate=sample_rate)


def angle_between(a, b, c):
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))))


def trunk_lean(shoulder, hip):
    torso = shoulder - hip
    denom = np.linalg.norm(torso)
    if denom < 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(torso, [0.0, -1.0]) / denom, -1, 1))))


def get_joints(landmarks, w, h):
    joints, norm = {}, {}
    for name, idx in LANDMARK_IDX.items():
        lm = landmarks[idx]
        if lm.visibility >= VIS_THRESHOLD:
            joints[name] = (int(lm.x * w), int(lm.y * h))
            norm[name]   = np.array([lm.x, lm.y])
    return joints, norm


def compute_angles(norm):
    angles = {}
    def pt(n): return norm.get(n)
    if all(k in norm for k in ["right_hip","right_knee","right_ankle"]):
        angles["knee"]       = angle_between(pt("right_hip"), pt("right_knee"), pt("right_ankle"))
    if all(k in norm for k in ["right_shoulder","right_hip","right_knee"]):
        angles["hip"]        = angle_between(pt("right_shoulder"), pt("right_hip"), pt("right_knee"))
    if all(k in norm for k in ["right_shoulder","right_elbow","right_wrist"]):
        angles["elbow"]      = angle_between(pt("right_shoulder"), pt("right_elbow"), pt("right_wrist"))
    if all(k in norm for k in ["right_hip","right_shoulder","right_elbow"]):
        angles["shoulder"]   = angle_between(pt("right_hip"), pt("right_shoulder"), pt("right_elbow"))
    if all(k in norm for k in ["right_shoulder","right_hip"]):
        angles["trunk lean"] = trunk_lean(pt("right_shoulder"), pt("right_hip"))
    return angles


def extract_features(rep_buffer, exercise):
    angle_map = {
        "knee":"knee_angle","hip":"hip_angle",
        "elbow":"elbow_angle","shoulder":"shoulder_angle","trunk lean":"angle_torso",
    }
    col_map = {}
    for short, long in angle_map.items():
        vals = [f[short] for f in rep_buffer if f.get(short) is not None]
        if vals:
            col_map[f"{long}_min"]   = np.min(vals)
            col_map[f"{long}_max"]   = np.max(vals)
            col_map[f"{long}_mean"]  = np.mean(vals)
            col_map[f"{long}_std"]   = np.std(vals)
            col_map[f"{long}_range"] = np.max(vals) - np.min(vals)
        else:
            for s in ["min","max","mean","std","range"]:
                col_map[f"{long}_{s}"] = None
    return {feat: col_map.get(feat) for feat in FEATURES[exercise]}


def annotate_frame(frame, joints, angles, last_result, in_rep):
    h, w = frame.shape[:2]

    for a, b in CONNECTIONS:
        if a in joints and b in joints:
            color = (0, 220, 0) if "right" in a else (180, 180, 180)
            cv2.line(frame, joints[a], joints[b], color, 2, cv2.LINE_AA)
    for px in joints.values():
        cv2.circle(frame, px, 5, (255, 255, 255), -1, cv2.LINE_AA)

    y = 30
    for label, val in angles.items():
        if val is not None:
            cv2.putText(frame, f"{label}: {val:.1f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
            y += 22

    if last_result:
        text  = "GOOD FORM" if last_result == "good" else "BAD FORM"
        color = (0, 200, 0) if last_result == "good" else (0, 0, 220)
        tw    = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0][0]
        cv2.putText(frame, text, ((w - tw) // 2, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)

    if in_rep:
        cv2.circle(frame, (w - 20, 20), 8, (0, 100, 255), -1, cv2.LINE_AA)


def webcam_loop(classifiers: dict) -> None:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap        = cv2.VideoCapture(1)
    start_time = time.time()
    in_rep     = False
    rep_buffer = []
    prev_ex    = None

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            timestamp_ms = int((time.time() - start_time) * 1000)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect_for_video(mp_image, timestamp_ms)

            with state.lock:
                exercise    = state.exercise
                last_result = state.last_result

            # Reset rep state when exercise changes
            if exercise != prev_ex:
                in_rep, rep_buffer = False, []
                prev_ex = exercise

            angles, joints = {}, {}
            if result.pose_landmarks:
                h, w   = frame.shape[:2]
                joints, norm = get_joints(result.pose_landmarks[0], w, h)
                angles = compute_angles(norm)

            if exercise and exercise in REP_CONFIG:
                cfg        = REP_CONFIG[exercise]
                signal_val = angles.get(cfg["signal"])
                mode       = cfg["mode"]

                if signal_val is not None:
                    if not in_rep:
                        if (mode == "valley" and signal_val < cfg["start_thresh"]) or \
                           (mode == "peak"   and signal_val > cfg["start_thresh"]):
                            in_rep, rep_buffer = True, []

                    if in_rep:
                        rep_buffer.append(angles)
                        if (mode == "valley" and signal_val > cfg["end_thresh"]) or \
                           (mode == "peak"   and signal_val < cfg["end_thresh"]):
                            if len(rep_buffer) > 15:
                                features = extract_features(rep_buffer, exercise)
                                if all(v is not None for v in features.values()):
                                    row        = [[features[f] for f in FEATURES[exercise]]]
                                    prediction = classifiers[exercise].predict(row)[0]
                                    with state.lock:
                                        state.last_result = prediction
                                        state.rep_count  += 1
                                    last_result = prediction
                                    if prediction == "good":
                                        play_ping()
                            in_rep, rep_buffer = False, []

            annotate_frame(frame, joints, angles, last_result, in_rep)

            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with state.lock:
                state.frame_jpg = jpg.tobytes()
                state.in_rep    = in_rep

    cap.release()


app = Flask(__name__)
LOCAL_IP = get_local_ip()


@app.route("/")
def index():
    return render_template("index.html", network_ip=LOCAL_IP)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with state.lock:
                jpg = state.frame_jpg
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/set_exercise", methods=["POST"])
def set_exercise():
    ex = request.get_json().get("exercise")
    if ex in REP_CONFIG:
        with state.lock:
            state.exercise    = ex
            state.rep_count   = 0
            state.last_result = None
    return jsonify(ok=True)


@app.route("/reset", methods=["POST"])
def reset():
    with state.lock:
        state.rep_count   = 0
        state.last_result = None
    return jsonify(ok=True)


@app.route("/status")
def status():
    with state.lock:
        return jsonify(
            exercise=state.exercise,
            reps=state.rep_count,
            result=state.last_result,
            in_rep=state.in_rep,
        )


if __name__ == "__main__":
    classifiers = {}
    for ex in ["squat", "ohp", "dlift"]:
        path = MODELS_DIR / f"{ex}_classifier.joblib"
        if path.exists():
            classifiers[ex] = joblib.load(path)
            print(f"Loaded {ex} classifier")
        else:
            print(f"WARNING: {ex} classifier not found — run train.py first")

    t = threading.Thread(target=webcam_loop, args=(classifiers,), daemon=True)
    t.start()

    print(f"\n  Local:   http://localhost:5000")
    print(f"  Network: http://{LOCAL_IP}:5000\n")

    # 0.0.0.0 makes it accessible on both localhost and network
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)