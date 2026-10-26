# Exercise Form Classifier

**Real-time feedback on barbell form from a webcam: after every rep, the app tells you whether it was good or bad.**

Machine Vision course project (CS 4885), University of Idaho, 2026 · Solo

<!-- TODO: add a short GIF of a squat with the good/bad result appearing -->

---

Point a camera at yourself, pick an exercise (**squat**, **overhead press**, or **deadlift**), and lift. After each rep you hear a high ping for good form or a double buzz for bad form, and the rep counter only counts clean reps.

| Exercise | What "bad form" means |
|---|---|
| Squat | Not reaching 90° of knee flexion (a shallow squat) |
| Overhead press | No lockout at the top, the elbow never fully extends |
| Deadlift | Rounded lower back, with increased trunk lean through the lift |

## How it works

```
Webcam → MediaPipe Pose → joint angles → rep segmentation → Random Forest → GOOD / BAD
```

Each rep is recorded as a time series of joint angles. When the rep ends, five statistics per angle (min, max, mean, standard deviation, range) go into a Random Forest trained for that exercise. Reps are found with peak detection on the angle signal.

The dataset is my own: 18 videos (3 good and 3 bad per exercise), segmented into **339 labelled reps**.

---

## The most important thing I got wrong, and fixed

My first evaluation split the 339 reps **randomly** into training and test sets. The accuracy looked excellent, and that was the problem.

Reps from the same video are nearly identical: same lighting, same camera angle, same bar, same session. With a random split, almost every test rep had a near-duplicate "sibling" in the training set, so the model was being tested on data it had effectively already seen. That is **data leakage**, and it inflates accuracy.

**The fix:** **leave-one-video-out cross-validation.** Every video is held out in turn, so the model is always tested on a recording it has never seen any part of.

| Exercise | Accuracy (leave-one-video-out) |
|---|---|
| Squat | 100% (121 / 121) |
| Overhead press | 99% (104 / 105) |
| Deadlift | 100% (113 / 113) |

The single miss is a bad overhead press classified as good, which makes sense: that fault is only about a 10° difference in elbow extension.

### What these numbers do *not* show

Every video is of **one person: me.** Leave-one-video-out proves the model generalizes to new recordings of me, not to other people with different body proportions. Testing that would need a multi-person dataset evaluated with **leave-one-subject-out** cross-validation, which is the next step.

---

## Other limitations

- **Side view only:** the lifter's right side must face the camera.
- **Rep-detection thresholds** are tuned to my movement and may need adjusting for others.
- **No authentication:** the Flask app is reachable by any device on the local network.

**Stack:** Python · MediaPipe · scikit-learn · OpenCV · SciPy · Flask

<details>
<summary>Running locally</summary>

Requires Python 3.10+, a webcam, and [`pose_landmarker_full.task`](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) in the project root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py   # open http://localhost:5000
```

To retrain from the original videos, run the scripts in `scripts/` in order: `extract_poses` → `calculate_angles` → `get_reps` → `build_dataset` → `train`.

</details>
