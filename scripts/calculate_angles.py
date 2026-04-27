import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR      = Path(__file__).parent.parent
LANDMARKS_DIR = BASE_DIR / "data/landmarks"
ANGLES_DIR    = BASE_DIR / "data/angles"

VIS_THRESHOLD = 0.7  # frames below this are treated as not visible

# Right side only (camera always sees right side)
REQUIRED_LANDMARKS = {
    "squat": ["right_shoulder", "right_hip", "right_knee", "right_ankle"],
    "ohp":   ["right_shoulder", "right_elbow", "right_wrist", "right_hip"],
    "dlift": ["right_shoulder", "right_hip", "right_knee", "right_ankle"],
}


def group_points(row: pd.Series, name: str) -> np.ndarray:
    return np.array([row[f"{name}_x"], row[f"{name}_y"]], dtype=float)


# Takes 3 joint positions and returns the angle between them
def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # angle in degrees at vertex b between points a-b-c
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8: # almost zero (0 not reliable with floats)
        return None
    cos_val = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


# Measures the angle between the torso and shoulder (can't use angle_between) only need 2 points
def angle_torso(shoulder: np.ndarray, hip: np.ndarray) -> float:
    # angle of torso from vertical in degrees -> 0 = upright, 90 = horizontal
    torso = shoulder - hip
    vertical = np.array([0.0, -1.0])  # up direction in image coords (y increases downward)
    denom = np.linalg.norm(torso)
    if denom < 1e-8: # almost zero (0 not reliable with floats)
        return None
    cos_val = np.dot(torso, vertical) / denom
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


# Check if visibility is above threshold
def all_visible(row: pd.Series, landmarks: list) -> bool:
    return all(row.get(f"{lm}_vis", 0) >= VIS_THRESHOLD for lm in landmarks)


def compute_row(row: pd.Series, exercise: str) -> dict:
    required = REQUIRED_LANDMARKS[exercise]

    if not all_visible(row, required): # return None if not visible
        if exercise == "squat":
            return {"knee_angle": None, "hip_angle": None, "angle_torso": None}
        elif exercise == "ohp":
            return {"elbow_angle": None, "shoulder_angle": None}
        elif exercise == "dlift":
            return {"hip_angle": None, "knee_angle": None, "angle_torso": None}

    if exercise == "squat":
        return {
            "knee_angle": angle_between(
                group_points(row, "right_hip"), group_points(row, "right_knee"), group_points(row, "right_ankle")
            ),
            "hip_angle": angle_between(
                group_points(row, "right_shoulder"), group_points(row, "right_hip"), group_points(row, "right_knee")
            ),
            "angle_torso": angle_torso(group_points(row, "right_shoulder"), group_points(row, "right_hip")),
        }

    if exercise == "ohp":
        return {
            "elbow_angle": angle_between(
                group_points(row, "right_shoulder"), group_points(row, "right_elbow"), group_points(row, "right_wrist")
            ),
            "shoulder_angle": angle_between(
                group_points(row, "right_hip"), group_points(row, "right_shoulder"), group_points(row, "right_elbow")
            ),
        }

    if exercise == "dlift":
        return {
            "hip_angle": angle_between(
                group_points(row, "right_shoulder"), group_points(row, "right_hip"), group_points(row, "right_knee")
            ),
            "knee_angle": angle_between(
                group_points(row, "right_hip"), group_points(row, "right_knee"), group_points(row, "right_ankle")
            ),
            "angle_torso": angle_torso(group_points(row, "right_shoulder"), group_points(row, "right_hip")),
        }


# Extract exercise type from filename ("squat_good_side_01" → "squat")
def process(path: Path) -> None:
    exercise = path.stem.split("_")[0]

    if exercise not in REQUIRED_LANDMARKS:
        print(f"Skipping {path.name} — unknown exercise '{exercise}'")
        return

    df = pd.read_csv(path)
    angle_rows = df.apply(lambda row: compute_row(row, exercise), axis=1, result_type="expand")
    out = pd.concat([df[["frame", "timestamp"]], angle_rows], axis=1)

    out_path = ANGLES_DIR / path.name
    out.to_csv(out_path, index=False)

    valid = out.iloc[:, 2].notna().sum()  # count not None rows using first angle column
    print(f"  {path.name}: {len(out)} frames, {valid} valid ({valid/len(out)*100:.0f}%)")


def main() -> None:
    ANGLES_DIR.mkdir(parents=True, exist_ok=True)

    csvs = sorted(LANDMARKS_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {LANDMARKS_DIR} —> run extract_poses.py first")
        return

    print(f"Found {len(csvs)} landmark files\n")
    for path in csvs:
        print(f"Processing {path.name}...")
        process(path)

    print(f"\nDone. Angle CSVs saved to {ANGLES_DIR}/")


if __name__ == "__main__":
    main()