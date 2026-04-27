import pandas as pd
from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent
REPS_DIR  = BASE_DIR / "data/reps"
OUT_FILE  = BASE_DIR / "data/dataset.csv"

# Angle columns present for each exercise
ANGLE_COLS = {
    "squat": ["knee_angle", "hip_angle", "angle_torso"],
    "ohp":   ["elbow_angle", "shoulder_angle"],
    "dlift": ["hip_angle", "knee_angle", "angle_torso"],
}


# Collapse a rep's angle time series into one row of summary statistics
def summarize(rep_df: pd.DataFrame, cols: list) -> dict:
    row = {}
    for col in cols:
        series = rep_df[col].dropna()
        if series.empty:
            row[f"{col}_min"]  = None
            row[f"{col}_max"]  = None
            row[f"{col}_mean"] = None
            row[f"{col}_std"]  = None
            row[f"{col}_range"] = None
        else:
            row[f"{col}_min"]   = round(series.min(), 2)
            row[f"{col}_max"]   = round(series.max(), 2)
            row[f"{col}_mean"]  = round(series.mean(), 2)
            row[f"{col}_std"]   = round(series.std(), 2)
            row[f"{col}_range"] = round(series.max() - series.min(), 2)
    return row


# Extract exercise and label from filename ('squat_good_side_01_rep_03' → ('squat', 'good'))
def parse_label(stem: str) -> tuple[str, str]:
    parts = stem.split("_")
    exercise = parts[0]   # squat / ohp / dlift
    label    = parts[1]   # good / bad
    return exercise, label


def main() -> None:
    rep_files = sorted(REPS_DIR.glob("*.csv"))
    if not rep_files:
        print(f"No rep CSVs found in {REPS_DIR} —> run get_reps.py first")
        return

    print(f"Found {len(rep_files)} rep files\n")

    rows = []
    skipped = 0

    for path in rep_files:
        exercise, label = parse_label(path.stem)

        if exercise not in ANGLE_COLS:
            print(f"Skipping {path.name} — unknown exercise '{exercise}'")
            skipped += 1
            continue

        rep_df = pd.read_csv(path)
        cols   = ANGLE_COLS[exercise]

        # skip reps where all angle data is missing
        if rep_df[cols].isna().all().all():
            print(f"Skipping {path.name} — no valid angle data")
            skipped += 1
            continue

        summary = summarize(rep_df, cols)
        summary["exercise"] = exercise
        summary["label"]    = label
        summary["source"]   = path.name
        rows.append(summary)

    if not rows:
        print("No valid reps found. Check that get_reps.py ran correctly.")
        return

    df = pd.DataFrame(rows)

    # put exercise and label columns first
    front = ["exercise", "label", "source"]
    df    = df[front + [c for c in df.columns if c not in front]]

    df.to_csv(OUT_FILE, index=False)

    print(f"Dataset saved to {OUT_FILE}")
    print(f"- Rows:    {len(df)}")
    print(f"- Skipped: {skipped}")
    print(f"\nClass distribution:")
    print(df.groupby(["exercise", "label"]).size().to_string())


if __name__ == "__main__":
    main()
