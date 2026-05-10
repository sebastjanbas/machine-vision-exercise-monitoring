import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "pose_landmarker_full.task"

CONNECTIONS = [
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("right_shoulder", "right_hip"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("left_shoulder",  "right_shoulder"),
    ("left_hip",       "right_hip"),
    ("left_shoulder",  "left_elbow"),
    ("left_hip",       "left_knee"),
]

LANDMARK_IDX = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow":    13, "right_elbow":    14,
    "left_wrist":    15, "right_wrist":    16,
    "left_hip":      23, "right_hip":      24,
    "left_knee":     25, "right_knee":     26,
    "left_ankle":    27, "right_ankle":    28,
}

RIGHT_COLOR = (0, 220, 0)
LEFT_COLOR  = (180, 180, 180)
JOINT_COLOR = (255, 255, 255)
TEXT_COLOR  = (0, 220, 255)
VIS_THRESHOLD = 0.5

BaseOptions          = mp.tasks.BaseOptions
PoseLandmarker       = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode          = mp.tasks.vision.RunningMode


def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return None
    cos_val = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def trunk_lean(shoulder, hip):
    torso = shoulder - hip
    vertical = np.array([0.0, -1.0])
    denom = np.linalg.norm(torso)
    if denom < 1e-8:
        return None
    cos_val = np.dot(torso, vertical) / denom
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def get_joints(landmarks, w, h):
    """Extract visible joint pixel positions from MediaPipe result."""
    joints = {}
    norm = {}  # normalized coords for angle math
    for name, idx in LANDMARK_IDX.items():
        lm = landmarks[idx]
        if lm.visibility >= VIS_THRESHOLD:
            joints[name] = (int(lm.x * w), int(lm.y * h))
            norm[name] = np.array([lm.x, lm.y])
    return joints, norm


def compute_angles(norm):
    angles = {}
    def pt(name):
        return norm.get(name)

    if all(k in norm for k in ["right_hip", "right_knee", "right_ankle"]):
        angles["knee"] = angle_between(pt("right_hip"), pt("right_knee"), pt("right_ankle"))
    if all(k in norm for k in ["right_shoulder", "right_hip", "right_knee"]):
        angles["hip"] = angle_between(pt("right_shoulder"), pt("right_hip"), pt("right_knee"))
    if all(k in norm for k in ["right_shoulder", "right_elbow", "right_wrist"]):
        angles["elbow"] = angle_between(pt("right_shoulder"), pt("right_elbow"), pt("right_wrist"))
    if all(k in norm for k in ["right_hip", "right_shoulder", "right_elbow"]):
        angles["shoulder"] = angle_between(pt("right_hip"), pt("right_shoulder"), pt("right_elbow"))
    if all(k in norm for k in ["right_shoulder", "right_hip"]):
        angles["trunk lean"] = trunk_lean(pt("right_shoulder"), pt("right_hip"))

    return angles


def draw(frame, joints, angles):
    h, w = frame.shape[:2]

    for a, b in CONNECTIONS:
        if a in joints and b in joints:
            color = RIGHT_COLOR if "right" in a else LEFT_COLOR
            cv2.line(frame, joints[a], joints[b], color, 2, cv2.LINE_AA)

    for name, px in joints.items():
        cv2.circle(frame, px, 5, JOINT_COLOR, -1, cv2.LINE_AA)

    # Angle labels in top-left panel
    y = 30
    for label, val in angles.items():
        if val is not None:
            text = f"{label}: {val:.1f}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, TEXT_COLOR, 1, cv2.LINE_AA)
            y += 25

    cv2.putText(frame, "Press Q to quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    print("Webcam open — stand sideways (right side toward camera). Press Q to quit.")

    start_time = time.time()

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((time.time() - start_time) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                h, w = frame.shape[:2]
                joints, norm = get_joints(result.pose_landmarks[0], w, h)
                angles = compute_angles(norm)
                draw(frame, joints, angles)

            cv2.imshow("Exercise Form Check", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()