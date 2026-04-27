import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RAW_DIR = ROOT_DIR / "data/raw"
OUT_DIR = ROOT_DIR / "data/landmarks"
MODEL_PATH = ROOT_DIR / "pose_landmarker_full.task"


# Only the joints needed for squat, deadlift, and OHP
# Indices are the same as the original BlazePose 33-landmark model
LANDMARKS = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
}

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def extract(video_path: Path, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,  # video mode uses tracking across frames
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows = []
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert color to rgb
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row = {"frame": frame_idx, "timestamp": round(frame_idx / fps, 4)}

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]  # first (only) detected person
                for name, idx in LANDMARKS.items():
                    lm = landmarks[idx]
                    row[f"{name}_x"]   = round(lm.x, 4)
                    row[f"{name}_y"]   = round(lm.y, 4)
                    row[f"{name}_vis"] = round(lm.visibility, 4)
            else:
                # No pose detected — fill with NaN so the frame is still recorded
                for name in LANDMARKS:
                    row[f"{name}_x"]   = None
                    row[f"{name}_y"]   = None
                    row[f"{name}_vis"] = 0.0

            rows.append(row)
            frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  {video_path.name}: {frame_idx} frames → {out_path.name}")


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Match both .mov and .MOV
    videos = sorted(RAW_DIR.glob("*.MOV")) + sorted(RAW_DIR.glob("*.mov"))

    if not videos:
        print(f"No videos found in {RAW_DIR}")
        return

    print(f"Found {len(videos)} videos\n")
    for video_path in videos:
        out_path = OUT_DIR / (video_path.stem + ".csv")
        print(f"Processing {video_path.name}...")
        extract(video_path, out_path)

    print(f"\nDone. CSVs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()