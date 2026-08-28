# Exercise Form Classifier

Real-time exercise form feedback using your webcam. Point a camera at yourself, select an exercise, and the app tells you after each rep whether your form was **good** or **bad** — with a sound cue and a rep counter that only increments for clean reps.

Supports three exercises: **squat**, **overhead press (OHP)**, and **deadlift**.

---

## Demo


---

## How It Works

```
Webcam → MediaPipe Pose → Joint Angles → Rep Segmentation → Random Forest → GOOD / BAD
```

Each rep is captured as a time-series of joint angles. When the rep completes, five summary statistics (min, max, mean, std, range) are extracted per angle and fed into a trained Random Forest classifier. The result is displayed on the web UI and played back as audio.

### Training Pipeline

The classifiers were trained on 18 videos (3 good + 3 bad per exercise, ~20 reps each):

| Script | Input → Output |
|---|---|
| `scripts/extract_poses.py` | Raw videos → landmark CSVs (MediaPipe BlazePose) |
| `scripts/calculate_angles.py` | Landmarks → joint angle CSVs |
| `scripts/get_reps.py` | Angle time-series → individual rep CSVs |
| `scripts/build_dataset.py` | Per-rep CSVs → `data/dataset.csv` (339 rows) |
| `scripts/train.py` | Dataset → `models/*.joblib` (Random Forest per exercise) |

### What "Bad Form" Means

| Exercise | Bad Form Definition |
|---|---|
| Squat | Not reaching 90° knee flexion (shallow squat) |
| OHP | No lockout at the top — elbow doesn't fully extend |
| Deadlift | Rounded lower back — increased trunk lean throughout lift |

### Classifier Accuracy (Leave-One-Video-Out CV)

| Exercise | Accuracy |
|---|---|
| Squat | 100% (121/121) |
| OHP | 99% (104/105) |
| Deadlift | 100% (113/113) |

The single OHP miss is a bad-form rep classified as good — expected given the subtlety of the fault (~10° difference in elbow extension).

---

## Web UI

```
┌─────────────────────────────────────────────┐
│           Exercise Form Classifier           │
├─────────────────────────────────────────────┤
│        [Squat]   [OHP]   [Deadlift]         │
├──────────────┬──────────────────────────────┤
│ Angles       │                              │
│ Knee   —     │       Live webcam feed       │
│ Hip    —     │                              │
│ Elbow  —     │   ● in-rep indicator         │
│ Shoulder —   │                              │
│ Trunk lean — │                              │
├──────────────┴──────────────────────────────┘
│     Reps: 0          Last Rep: —            │
│              [Reset Counter]                │
└─────────────────────────────────────────────┘
```

- Live joint angles update at 400 ms intervals
- Rep counter only increments for good-form reps
- Audio feedback: high ping = good form, double buzz = bad form
- Orange dot in the top-right corner of the video feed while a rep is in progress

---

## Quick Start

### Requirements

- Python 3.10+
- A webcam
- [`pose_landmarker_full.task`](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) placed in the project root

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install sounddevice  # not in requirements.txt but required for audio
```

### Run (models are already trained)

```bash
python app.py
```

Open `http://localhost:5000`, select an exercise, and stand so your **right side faces the camera**.

### Retrain from Scratch

Run the pipeline scripts in order (requires the original `.MOV` video files):

```bash
python scripts/extract_poses.py
python scripts/calculate_angles.py
python scripts/get_reps.py
python scripts/build_dataset.py
python scripts/train.py
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Pose estimation | MediaPipe Pose Landmarker (BlazePose) |
| Classifier | Random Forest (scikit-learn) |
| Web server | Flask |
| Video capture | OpenCV |
| Audio feedback | sounddevice + numpy (synthesized sine waves) |
| Rep segmentation | scipy `find_peaks` |

---

## Limitations

- **Single subject** — trained on one person. May need retraining for other individuals.
- **Right-side only** — the subject must face left (right side toward the camera).
- **Fixed thresholds** — rep detection uses angle thresholds tuned for the training subject; may need adjustment for different body proportions.
- **No authentication** — the Flask app is accessible to any device on the local network.
