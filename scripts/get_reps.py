import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
ANGLES_DIR = BASE_DIR / "data/angles"
REPS_DIR   = BASE_DIR / "data/reps"

# Which angle to use for rep detection and whether to find peaks or valleys
REP_SIGNAL = {
    "squat": ("knee_angle",   "valley"),  # knee angle dips at bottom of squat
    "ohp":   ("elbow_angle",  "peak"),    # elbow angle peaks at top of press
    "dlift": ("hip_angle",    "peak"),    # hip angle peaks at top of deadlift
}

SMOOTH_WINDOW = 15   # frames to average for smoothing (~0.5s at 30fps)
MIN_REP_FRAMES = 20  # minimum frames a rep must span to be counted (~0.7s)

# Rolling average to reduce between frame noise
def smooth(signal: np.ndarray) -> np.ndarray:
    return pd.Series(signal).rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().to_numpy()


def find_rep_boundaries(signal: np.ndarray, mode: str) -> np.ndarray:
    # return indices of rep turning points (valleys or peaks)
    if mode == "valley":
        indices, _ = find_peaks(-signal, distance=MIN_REP_FRAMES)
    else:
        indices, _ = find_peaks(signal, distance=MIN_REP_FRAMES)
    return indices


def segment(path: Path) -> int:
    exercise = path.stem.split("_")[0]

    if exercise not in REP_SIGNAL:
        print(f"Skipping {path.name} — unknown exercise '{exercise}'")
        return 0

    signal_col, mode = REP_SIGNAL[exercise]
    df = pd.read_csv(path)

    # only the rows where there is a valid angle measurement
    valid = df[df[signal_col].notna()].copy()
    valid = valid.reset_index(drop=True)

    if len(valid) < MIN_REP_FRAMES * 3:
        print(f"{path.name}: not enough valid frames ({len(valid)}), skipping")
        return 0

    signal = smooth(valid[signal_col].to_numpy())
    turning_points = find_rep_boundaries(signal, mode)

    if len(turning_points) < 2:
        print(f"{path.name}: fewer than 2 turning points found, skipping")
        return 0

    # each rep runs from one turning point to the next
    rep_count = 0
    for i in range(len(turning_points) - 1):
        start_idx = turning_points[i]
        end_idx   = turning_points[i + 1]

        rep_df = valid.iloc[start_idx:end_idx + 1].copy()

        rep_num  = rep_count + 1
        out_name = f"{path.stem}_rep_{rep_num:02d}.csv"
        rep_df.to_csv(REPS_DIR / out_name, index=False)
        rep_count += 1

    return rep_count


def main() -> None:
    REPS_DIR.mkdir(parents=True, exist_ok=True)

    csvs = sorted(ANGLES_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {ANGLES_DIR} —> run calculate_angles.py first")
        return

    print(f"Found {len(csvs)} angle files\n")
    total_reps = 0
    for path in csvs:
        print(f"Processing {path.name}...")
        reps = segment(path)
        print(f"  -> {reps} reps saved")
        total_reps += reps

    print(f"\nDone. {total_reps} rep CSVs saved to {REPS_DIR}/")
    print(f"Expected {len(csvs) * 20} reps ({len(csvs)} videos x 20 reps)")


if __name__ == "__main__":
    main()