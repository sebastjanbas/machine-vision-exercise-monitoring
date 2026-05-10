import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR   = Path(__file__).parent.parent
DATASET    = BASE_DIR / "data/dataset.csv"
MODELS_DIR = BASE_DIR / "models"

# Feature columns for each exercise (must match build_dataset.py ANGLE_COLS)
FEATURES = {
    "squat": [
        "knee_angle_min", "knee_angle_max", "knee_angle_mean", "knee_angle_std", "knee_angle_range",
        "hip_angle_min",  "hip_angle_max",  "hip_angle_mean",  "hip_angle_std",  "hip_angle_range",
        "angle_torso_min","angle_torso_max","angle_torso_mean","angle_torso_std","angle_torso_range",
    ],
    "ohp": [
        "elbow_angle_min",    "elbow_angle_max",    "elbow_angle_mean",    "elbow_angle_std",    "elbow_angle_range",
        "shoulder_angle_min", "shoulder_angle_max", "shoulder_angle_mean", "shoulder_angle_std", "shoulder_angle_range",
    ],
    "dlift": [
        "hip_angle_min",  "hip_angle_max",  "hip_angle_mean",  "hip_angle_std",  "hip_angle_range",
        "knee_angle_min", "knee_angle_max", "knee_angle_mean", "knee_angle_std", "knee_angle_range",
        "angle_torso_min","angle_torso_max","angle_torso_mean","angle_torso_std","angle_torso_range",
    ],
}

# leave one video out
# each of the 6 videos takes a turn as the test set while the other 5 train.
# returns all true labels and predictions collected across all 6 folds.
def leave_one_video_out(sub: pd.DataFrame, feature_cols: list) -> tuple:
    # Source filename without _rep_XX suffix identifies the video
    sub = sub.copy()
    sub["video"] = sub["source"].str.replace(r"_rep_\d+\.csv$", "", regex=True)
    videos = sub["video"].unique()

    all_true = []
    all_pred = []

    for held_out in videos:
        train = sub[sub["video"] != held_out]
        test  = sub[sub["video"] == held_out]

        X_train = train[feature_cols]
        y_train = train["label"]
        X_test  = test[feature_cols]
        y_test  = test["label"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        all_true.extend(y_test.tolist())
        all_pred.extend(model.predict(X_test).tolist())

    return all_true, all_pred


def train_exercise(df: pd.DataFrame, exercise: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {exercise.upper()}")
    print(f"{'='*50}")

    sub = df[df["exercise"] == exercise].copy()
    feature_cols = FEATURES[exercise]

    print(f"Samples: {len(sub)}  (good={sum(sub['label']=='good')}, bad={sum(sub['label']=='bad')})")

    # Leave-one-video-out evaluation — more honest than random rep-level split
    all_true, all_pred = leave_one_video_out(sub, feature_cols)

    correct = sum(t == p for t, p in zip(all_true, all_pred))
    print(f"\nLeave-one-video-out accuracy: {correct}/{len(all_true)} = {correct/len(all_true):.3f}")

    print(f"\nFull results ({len(all_true)} reps across 6 folds):")
    print(classification_report(all_true, all_pred, target_names=["bad", "good"]))

    print("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(all_true, all_pred, labels=["bad", "good"])
    print(f"           pred:bad  pred:good")
    print(f"actual:bad    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"actual:good   {cm[1,0]:4d}      {cm[1,1]:4d}")

    # Train final model on ALL data for deployment
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(sub[feature_cols], sub["label"])

    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop 5 most important features:")
    for feat, imp in importances[:5]:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<30} {imp:.3f}  {bar}")

    out_path = MODELS_DIR / f"{exercise}_classifier.joblib"
    joblib.dump(model, out_path)
    print(f"\nModel saved to {out_path}")


def main() -> None:
    if not DATASET.exists():
        print(f"dataset.csv not found — run build_dataset.py first")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET)

    for exercise in ["squat", "ohp", "dlift"]:
        train_exercise(df, exercise)

    print("\n\nAll models trained and saved to models/")


if __name__ == "__main__":
    main()