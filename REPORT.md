# Exercise Form Classifier — Project Report

## Overview

This project is a real-time exercise form classifier that uses a webcam, MediaPipe Pose, and a trained Random Forest classifier to detect whether a person is performing a squat, overhead press (OHP), or deadlift with correct or incorrect form. It runs as a local Flask web application, streams annotated video to the browser, displays live joint angles, counts correctly-executed reps, and plays audio feedback after each rep.

---

## Repository Structure

```
ExerciseDetection/
├── app.py                          # Flask web application + webcam loop
├── requirements.txt                # Python dependencies
├── pose_landmarker_full.task       # MediaPipe model binary (not in git)
├── NOTES.md                        # Issue log from development
├── REPORT.md                       # This file
│
├── scripts/
│   ├── extract_poses.py            # Stage 1: video → landmark CSVs
│   ├── calculate_angles.py         # Stage 2: landmarks → angle CSVs
│   ├── get_reps.py                 # Stage 3: angles → per-rep CSVs
│   ├── build_dataset.py            # Stage 4: reps → dataset.csv
│   └── train.py                    # Stage 5: dataset → trained models
│
├── data/
│   ├── landmarks/                  # Output of extract_poses.py (18 CSVs)
│   ├── angles/                     # Output of calculate_angles.py (18 CSVs)
│   ├── reps/                       # Output of get_reps.py (~339 CSVs)
│   └── dataset.csv                 # Output of build_dataset.py (339 rows)
│
├── models/
│   ├── squat_classifier.joblib     # Trained Random Forest for squats
│   ├── ohp_classifier.joblib       # Trained Random Forest for OHP
│   └── dlift_classifier.joblib     # Trained Random Forest for deadlifts
│
├── templates/
│   └── index.html                  # Web UI
└── static/
    └── style.css                   # Stylesheet
```

---

## Data Collection

### Videos
- **18 videos** total, split evenly across 3 exercises and 2 form classes:
  - Squat: 3 good-form videos, 3 bad-form videos
  - OHP: 3 good-form videos, 3 bad-form videos
  - Deadlift: 3 good-form videos, 3 bad-form videos
- Each video contains approximately 20 repetitions filmed from the right side.
- All videos were recorded by a single subject in a single session.
- Naming convention: `{exercise}_{label}_side_{nn}.MOV`  
  Example: `squat_good_side_01.MOV`

### Bad Form Definitions
- **Squat bad form**: Not reaching 90° knee flexion (shallow squat — knee angle stays above ~130°).
- **OHP bad form**: No lockout at the top — elbow not fully extended (~163° max vs ~173° for good form).
- **Deadlift bad form**: Rounded lower back — increased trunk lean throughout the lift.

---

## Pipeline: scripts/

### Stage 1 — `extract_poses.py`

Reads each raw video and runs **MediaPipe Pose Landmarker** (Tasks API, `pose_landmarker_full.task` model) in `VIDEO` running mode. For every frame, it records the normalized (x, y) coordinates and visibility score of 12 landmarks:

| Landmark | BlazePose Index |
|---|---|
| left_shoulder / right_shoulder | 11 / 12 |
| left_elbow / right_elbow | 13 / 14 |
| left_wrist / right_wrist | 15 / 16 |
| left_hip / right_hip | 23 / 24 |
| left_knee / right_knee | 25 / 26 |
| left_ankle / right_ankle | 27 / 28 |

Frames where no pose is detected are still recorded, with coordinate columns set to `None` and visibility set to `0.0`, so frame numbering stays intact.

**Output**: One CSV per video in `data/landmarks/`, with columns `frame`, `timestamp`, and `{name}_x`, `{name}_y`, `{name}_vis` for each of the 12 landmarks. Coordinates are in normalized [0, 1] image space.

**Key settings**:
- `RunningMode.VIDEO` — enables inter-frame tracking for smoother landmark positions
- `num_poses=1` — only one person in frame
- All three confidence thresholds set to `0.5`

---

### Stage 2 — `calculate_angles.py`

Reads each landmark CSV and computes joint angles frame by frame. Only the **right side** is used (camera consistently sees the right side of the subject). Frames where any required landmark has `visibility < 0.7` produce `None` for that exercise's angles.

**Angles computed per exercise**:

| Exercise | Angles |
|---|---|
| Squat | `knee_angle`, `hip_angle`, `angle_torso` |
| OHP | `elbow_angle`, `shoulder_angle` |
| Deadlift | `hip_angle`, `knee_angle`, `angle_torso` |

**Angle formulas**:

- `angle_between(a, b, c)` — angle at joint `b` between segments `b→a` and `b→c`, computed using the dot product:
  ```
  cos θ = (ba · bc) / (|ba| × |bc|)
  θ = arccos(clipped value)
  ```
  The `np.clip(-1, 1)` prevents domain errors from floating point drift. The `1e-8` zero-length guard prevents division by zero when two landmarks collapse to the same point.

- `angle_torso(shoulder, hip)` — angle of the torso segment from vertical. Uses only two points; the reference vector is `[0, -1]` (upward in image coordinates, since y increases downward). Output is 0° when standing perfectly upright and increases as the torso leans forward.

**Specific angle definitions**:
- `knee_angle`: hip → knee → ankle. Full extension = ~180°, deep squat bottom = ~35°.
- `hip_angle`: shoulder → hip → knee. Full extension = ~180°, hinge = ~60°.
- `elbow_angle`: shoulder → elbow → wrist. Full extension = ~180°, bent = ~90°.
- `shoulder_angle`: hip → shoulder → elbow. Arm at side = ~0°, overhead = ~180°.
- `angle_torso`: deviation from vertical. 0° = upright, increases with forward lean.

**Output**: One CSV per video in `data/angles/`, with columns `frame`, `timestamp`, and one column per angle (or `None` for invisible frames).

---

### Stage 3 — `get_reps.py`

Segments the continuous angle signal from each video into individual repetitions. Each exercise uses a specific signal and turning-point type:

| Exercise | Signal | Turning point |
|---|---|---|
| Squat | `knee_angle` | Valley (minimum at bottom of squat) |
| OHP | `elbow_angle` | Peak (maximum at top of press) |
| Deadlift | `hip_angle` | Peak (maximum at standing position) |

**Processing steps**:
1. **Filter** rows where the signal angle is `None` (landmark not visible).
2. **Smooth** the signal with a 15-frame rolling average (`SMOOTH_WINDOW = 15`, ~0.5 s at 30 fps) to reduce per-frame noise.
3. **Detect turning points** using `scipy.signal.find_peaks` with:
   - `distance = 45` (`MIN_REP_FRAMES`) — minimum 45 frames (~1.5 s) between consecutive peaks/valleys. Prevents detecting wobble within a single rep as multiple reps.
   - `prominence = 20` — a peak/valley must stand out by at least 20° from its surroundings. Filters noise that doesn't represent a genuine movement.
4. **Segment** the angle data between consecutive turning points: each segment from turning point `i` to turning point `i+1` is one rep.

**Output**: One CSV per rep in `data/reps/`, named `{video_stem}_rep_{nn}.csv`. Each CSV contains the angle time series for that single repetition.

**Tuning history**: The initial `MIN_REP_FRAMES = 20` produced 35–50 reps per video instead of the expected ~20. Raising it to 45 and adding `prominence = 20` brought the count to 339 reps across 18 videos (~18.8 per video). The shortfall from the expected 360 is attributed to the OHP bad-form videos where the flat elbow signal is harder to segment. See `NOTES.md` for details.

---

### Stage 4 — `build_dataset.py`

Collapses each per-rep time-series CSV into a single row of **summary statistics**. This converts variable-length angle sequences into a fixed-size feature vector suitable for a tabular classifier.

For each angle column in a rep, five statistics are computed:

| Statistic | Meaning |
|---|---|
| `_min` | Minimum angle value in the rep |
| `_max` | Maximum angle value in the rep |
| `_mean` | Mean angle across all frames |
| `_std` | Standard deviation (consistency of movement) |
| `_range` | max − min (total range of motion) |

Exercise and label are parsed from the filename (`squat_good_side_01_rep_03` → exercise=`squat`, label=`good`).

**Output**: `data/dataset.csv` with 339 rows and 28 columns (exercise, label, source, plus 25 feature columns).

---

## Dataset

### Statistics

| Exercise | Bad | Good | Total |
|---|---|---|---|
| Squat | 63 | 58 | 121 |
| OHP | 48 | 57 | 105 |
| Deadlift | 53 | 60 | 113 |
| **Total** | **164** | **175** | **339** |

Classes are well-balanced within each exercise (no more than ~10% imbalance). The dataset is modest in size and comes from a single subject.

### Feature Columns

**Squat** (15 features): `knee_angle_{min,max,mean,std,range}`, `hip_angle_{min,max,mean,std,range}`, `angle_torso_{min,max,mean,std,range}`

**OHP** (10 features): `elbow_angle_{min,max,mean,std,range}`, `shoulder_angle_{min,max,mean,std,range}`

**Deadlift** (15 features): `hip_angle_{min,max,mean,std,range}`, `knee_angle_{min,max,mean,std,range}`, `angle_torso_{min,max,mean,std,range}`

### Biomechanical separation between classes

The angle differences between good and bad form are large and consistent:

| Exercise | Key feature | Good form | Bad form |
|---|---|---|---|
| Squat | `knee_angle_min` | ~36.5° | ~131.6° |
| OHP | `elbow_angle_max` | ~173° | ~163° |
| Deadlift | `angle_torso_max` | lower | higher (rounded back) |

These large differences explain the high classification accuracy. Squat and deadlift have near-perfect separability; OHP is the most subtle and most likely to misclassify.

---

## Model Training — `train.py`

### Algorithm
**Random Forest Classifier** (scikit-learn) with 100 decision trees and `random_state=42`. No hyperparameter tuning was performed; the default settings were sufficient given the large feature separation.

### Evaluation: Leave-One-Video-Out (LOVO) Cross-Validation

Each exercise has 6 videos (3 good, 3 bad). LOVO trains on 5 videos and tests on the held-out 6th, rotating through all 6 combinations. This ensures the test set never contains reps from a video used during training.

This approach was necessary because a naive random 80/20 rep-level split resulted in 100% accuracy due to **data leakage**: reps from the same video (same person, same session) appeared in both train and test sets, so the model effectively memorized the subject's movement signature rather than learning a generalizable classifier. See `NOTES.md` for the full explanation.

### LOVO Results

| Exercise | Correct | Total | Accuracy |
|---|---|---|---|
| Squat | 121 | 121 | **100%** |
| OHP | 104 | 105 | **99%** |
| Deadlift | 113 | 113 | **100%** |

The single OHP misclassification (one bad-form rep classified as good) is expected given the subtlety of the no-lockout fault (~10° difference in elbow max angle). This confirms the evaluation is working honestly — a perfect score on all three would be suspicious.

### Deployment Model
After LOVO evaluation, a final model is trained on **all available data** for each exercise and saved to `models/{exercise}_classifier.joblib`. This model is what `app.py` loads at runtime.

### Feature Importances (approximate, from final models)
The most informative features are range-of-motion statistics — `_min`, `_max`, and `_range` — since these directly capture whether the subject reached the required depth or extension. Standard deviation and mean are less important but help with consistency signals.

---

## Real-Time Application — `app.py`

The application has two concurrent execution contexts:

### Architecture

```
Main thread (Flask)         Background thread (webcam_loop)
────────────────────        ────────────────────────────────
Serves HTTP requests   ←──  Writes to shared State object
/                           (frame_jpg, angles, in_rep,
/video_feed                  rep_count, last_result)
/set_exercise
/reset
/status
```

A `threading.Lock` inside the `State` class ensures only one thread accesses shared fields at a time, preventing race conditions.

### Shared State (`State` class)

| Field | Type | Description |
|---|---|---|
| `lock` | `threading.Lock` | Mutual exclusion for all other fields |
| `exercise` | `str \| None` | Currently selected exercise |
| `rep_count` | `int` | Number of good-form reps counted |
| `last_result` | `str \| None` | `"good"` or `"bad"` from most recent rep |
| `in_rep` | `bool` | Whether a rep is currently in progress |
| `frame_jpg` | `bytes \| None` | Latest JPEG-encoded frame |
| `angles` | `dict` | Latest computed angle values |

### Webcam Loop (`webcam_loop`)

Runs in a daemon thread. Each iteration:

1. **Capture** a frame from `cv2.VideoCapture(0)`.
2. **Run MediaPipe** in `VIDEO` mode with a monotonically increasing timestamp.
3. **Extract joints** — converts normalized landmark coordinates to pixel coordinates and numpy arrays. Landmarks with `visibility < 0.5` are excluded.
4. **Compute angles** — same formulas as `calculate_angles.py`, adapted for real-time use.
5. **Rep state machine**:
   - **Waiting**: monitors the signal angle. Enters `in_rep` state when the angle crosses `start_thresh` (valley mode: signal drops below threshold; peak mode: signal rises above threshold).
   - **Recording**: accumulates `angles` dicts into `rep_buffer` every frame.
   - **Completing**: when the signal crosses `end_thresh` in the return direction and `rep_buffer` has at least 15 frames, extracts features, runs the classifier, updates state, and plays audio.
6. **Annotate frame** — draws skeleton lines and joint dots. Right-side bones are bright green, left-side are grey. An orange dot appears in the top-right corner while a rep is in progress.
7. **Encode** the frame as JPEG (quality 80) and write to `state.frame_jpg`.

### Rep Detection Thresholds

| Exercise | Signal | Mode | Start threshold | End threshold |
|---|---|---|---|---|
| Squat | knee angle | valley | < 140° | > 155° |
| OHP | elbow angle | peak | > 120° | < 100° |
| Deadlift | hip angle | valley | < 150° | > 165° |

A rep must also accumulate at least 15 frames (~0.5 s) in the buffer before being classified, filtering out accidental triggers.

### Rep Counting Logic
- A rep is only counted if the classifier predicts `"good"`.
- Bad-form reps are classified and displayed in the UI but do not increment the counter.

### Audio Feedback

| Event | Sound |
|---|---|
| Good form | Single high-pitched ping (1050 Hz, 0.18 s, exponential fade) |
| Bad form | Two short low-pitched beeps (300 Hz, 0.15 s each, 70 ms gap) |

Sounds are synthesized in real time using `sounddevice` and `numpy` (sine wave with exponential amplitude envelope).

### HTTP Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves `index.html` |
| `/video_feed` | GET | MJPEG stream (multipart/x-mixed-replace) at ~30 fps |
| `/set_exercise` | POST | Sets active exercise, resets rep count and last result |
| `/reset` | POST | Resets rep count and last result only |
| `/status` | GET | Returns JSON: `exercise`, `reps`, `result`, `in_rep`, `angles` |

The server binds to `0.0.0.0:5000`, making it accessible to other devices on the local network.

---

## Frontend — `templates/index.html` + `static/style.css`

### Layout

```
┌─────────────────────────────────────────────┐
│           Exercise Form Classifier           │
├───────────────────────────────────────────┤
│        [Squat]   [OHP]   [Deadlift]         │
├──────────────┬──────────────────────────────┤
│ ANGLES       │                              │
│ Knee   —     │       WEBCAM STREAM          │
│ Hip    —     │       (1200 px wide)         │
│ Elbow  —     │                              │
│ Shoulder —   │    ● (blue dot = in-rep)     │
│ Trunk lean — │                              │
├──────────────┴──────────────────────────────┤
│     Reps: 0          Last Rep: —            │
│              [Reset Counter]                │
│        Select an exercise to begin          │
└─────────────────────────────────────────────┘
```

### JavaScript Behaviour
- **`setExercise(ex)`**: POSTs to `/set_exercise`, highlights the active button, resets the rep counter and form result display.
- **`resetReps()`**: POSTs to `/reset`, resets the counter display.
- **`setFormResult(result)`**: Updates the "Last Rep" box with colour-coded text (green = GOOD FORM ✓, red = BAD FORM ✗).
- **`updateAngles(angles)`**: Iterates over the five angle keys and updates each row in the angle panel. Displays `—` when the value is null (joint not visible).
- **Polling**: `setInterval` fires every 400 ms, fetches `/status`, and updates the rep counter, form result, in-rep indicator, and all angle values.

### Visual Design
- Dark theme: `#0f0f1a` background.
- Active exercise button: cyan border (`#00b4d8`) with dark teal background.
- Rep counter: large cyan `#00e5ff` numerals.
- Form result:
  - Good: green (`#00e676`) with translucent green background.
  - Bad: red (`#ff1744`) with translucent red background.
  - None: dark grey.
- Angle panel: dark card (`#1a1a2e`) to the left of the video feed.
- In-rep indicator: blue pulsing dot overlaid on the top-right corner of the video, animated with a CSS keyframe.

---

## Dependencies — `requirements.txt`

| Package | Purpose |
|---|---|
| `mediapipe >= 0.10.14` | Pose landmark detection (BlazePose) |
| `opencv-python >= 4.10.0` | Video capture, frame drawing, JPEG encoding |
| `pandas >= 2.2.0` | CSV I/O and data manipulation throughout the pipeline |
| `numpy >= 1.26.0` | Angle computation, audio synthesis |
| `scikit-learn >= 1.5.0` | Random Forest classifier and evaluation utilities |
| `scipy` | `find_peaks` for rep segmentation |
| `joblib` | Model serialisation (save/load `.joblib` files) |
| `flask` | Web server and routing |
| `sounddevice` | Real-time audio output (not in requirements.txt — must be installed separately) |

Note: `sounddevice` is used in `app.py` but is missing from `requirements.txt`.

---

## How to Run

### First-time setup (training pipeline)
Run scripts in order. Each script depends on the output of the previous one.

```bash
python scripts/extract_poses.py    # raw videos → data/landmarks/
python scripts/calculate_angles.py # landmarks → data/angles/
python scripts/get_reps.py         # angles → data/reps/
python scripts/build_dataset.py    # reps → data/dataset.csv
python scripts/train.py            # dataset → models/*.joblib
```

### Running the application
```bash
python app.py
```
Open `http://localhost:5000` in a browser. Select an exercise and stand in frame.

---

## Known Issues and Limitations

### 1. Single Subject
All 339 training reps were recorded by one person. The classifiers learn that person's specific movement signature. Performance on other individuals is unknown and likely to be lower without retraining on their data.

### 2. Right-Side-Only
Both the training pipeline and the real-time app use only right-side landmarks. The subject must always face left (right side toward the camera).

### 3. Rep Over-Detection (Resolved)
Initial segmentation with `MIN_REP_FRAMES = 20` detected 35–50 reps per video instead of ~20, corrupting the training data. Resolved by raising `MIN_REP_FRAMES` to 45 and adding `prominence = 20`. All data was regenerated after the fix. Documented in `NOTES.md`.

### 4. Data Leakage in Initial Evaluation (Resolved)
An initial random 80/20 rep-level split produced 100% accuracy due to same-video reps appearing in both train and test. Resolved by switching to leave-one-video-out cross-validation. Documented in `NOTES.md`.

### 5. OHP Segmentation Shortfall
OHP bad-form videos (no-lockout fault) produce a flatter elbow angle signal, making peak detection harder. These videos yield fewer than 20 detected reps. This is a data limitation rather than a code bug.

### 6. Missing sounddevice in requirements.txt
`sounddevice` is imported in `app.py` but not listed in `requirements.txt`. Installing from the requirements file will cause an `ImportError` at startup.

### 7. Real-Time Rep Detection Sensitivity
The threshold-based state machine in `app.py` uses fixed angle thresholds. These were tuned for the subject who recorded the training data and may need adjustment for different body proportions or camera distances.

### 8. No Authentication or Multi-User Support
The Flask app has no authentication. Any device on the local network that knows the IP and port can access and control the exercise selection and reset the counter.