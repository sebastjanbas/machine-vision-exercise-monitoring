import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR      = Path(__file__).parent.parent
RAW_DIR       = BASE_DIR / "data/raw"
LANDMARKS_DIR = BASE_DIR / "data/landmarks"
ANGLES_DIR    = BASE_DIR / "data/angles"
OUT_DIR       = BASE_DIR / "data/output"

# Bones to draw — (joint_a, joint_b)
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

# Which joint to draw each angle label next to
ANGLE_LABEL_JOINT = {
    "knee_angle":    "right_knee",
    "hip_angle":     "right_hip",
    "elbow_angle":   "right_elbow",
    "shoulder_angle":"right_shoulder",
    "angle_torso":   "right_hip",
}

RIGHT_COLOR = (0, 220, 0)    # green — right side (facing camera)
LEFT_COLOR  = (180, 180, 180) # gray  — left side (away from camera)
JOINT_COLOR = (255, 255, 255) # white — joint dots
TEXT_COLOR  = (0, 220, 255)  # yellow — angle labels
VIS_THRESHOLD = 0.5


def to_px(x: float, y: float, w: int, h: int) -> tuple:
    """Convert normalized [0,1] coordinates to pixel coordinates."""
    return (int(x * w), int(y * h))


def draw_frame(frame: np.ndarray, lm_row: pd.Series, angle_row: pd.Series) -> np.ndarray:
    h, w = frame.shape[:2]

    # Collect joint pixel positions
    joints = {}
    for col in lm_row.index:
        if col.endswith("_x"):
            name = col[:-2]
            x = lm_row.get(f"{name}_x")
            y = lm_row.get(f"{name}_y")
            vis = lm_row.get(f"{name}_vis", 0)
            if pd.notna(x) and pd.notna(y) and vis >= VIS_THRESHOLD:
                joints[name] = to_px(float(x), float(y), w, h)

    # Draw bones
    for a, b in CONNECTIONS:
        if a in joints and b in joints:
            color = RIGHT_COLOR if "right" in a else LEFT_COLOR
            cv2.line(frame, joints[a], joints[b], color, 2, cv2.LINE_AA)

    # Draw joint dots
    for name, px in joints.items():
        cv2.circle(frame, px, 5, JOINT_COLOR, -1, cv2.LINE_AA)

    # Draw angle labels
    if angle_row is not None:
        for angle_col, joint_name in ANGLE_LABEL_JOINT.items():
            if angle_col in angle_row.index and joint_name in joints:
                val = angle_row[angle_col]
                if pd.notna(val):
                    label = f"{angle_col.replace('_', ' ')}: {val:.1f}"
                    px = joints[joint_name]
                    offset = (px[0] + 10, px[1] - 10)
                    cv2.putText(frame, label, offset, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return frame


def process(video_name: str) -> None:
    stem = Path(video_name).stem
    video_path = RAW_DIR / video_name
    lm_path    = LANDMARKS_DIR / (stem + ".csv")
    angle_path = ANGLES_DIR    / (stem + ".csv")

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return
    if not lm_path.exists():
        print(f"Landmarks not found: {lm_path} — run extract_poses.py first")
        return

    lm_df    = pd.read_csv(lm_path).set_index("frame")
    angle_df = pd.read_csv(angle_path).set_index("frame") if angle_path.exists() else None

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = OUT_DIR / (stem + "_annotated.mp4")
    writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm_row    = lm_df.loc[frame_idx]    if frame_idx in lm_df.index    else None
        angle_row = angle_df.loc[frame_idx] if (angle_df is not None and frame_idx in angle_df.index) else None

        if lm_row is not None:
            frame = draw_frame(frame, lm_row, angle_row)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Accept a specific video name as argument, otherwise process all
    if len(sys.argv) > 1:
        process(sys.argv[1])
    else:
        videos = sorted(RAW_DIR.glob("*.MOV")) + sorted(RAW_DIR.glob("*.mov"))
        print(f"Found {len(videos)} videos\n")
        for v in videos:
            print(f"Processing {v.name}...")
            process(v.name)


if __name__ == "__main__":
    main()
