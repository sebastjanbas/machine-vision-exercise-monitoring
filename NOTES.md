# Project Notes

## Issues and Limitations

### Rep Segmentation Over-Detection (Data Regeneration Required)
**Discovered:** After initial run of `get_reps.py`

**Problem:**
The initial `MIN_REP_FRAMES = 20` parameter (0.67 seconds minimum between turning points)
was too small. The rep segmentation script was detecting small wobbles and noise within a
single rep as separate reps, producing roughly double the expected count — approximately
35–50 reps per video instead of 20.

```
squat_good_side_03: 50 reps  (expected ~20)
squat_good_side_01: 43 reps  (expected ~20)
```

This corrupted the training data because each row in `dataset.csv` represents one rep.
A single squat split into 3 fragments means the statistical features (min/max/mean angle)
are computed over a third of a movement, not a full one.

**Fix:**
- Increased `MIN_REP_FRAMES` from 20 → 45 (~1.5 seconds at 30fps)
- Added `PROMINENCE = 20` to `find_peaks` so only genuine peaks/valleys that stand out
  by at least 20° are counted, filtering out small signal noise
- Deleted all rep CSVs in `data/reps/` and re-ran `get_reps.py` and `build_dataset.py`

**Result after fix:** 339 reps across 18 videos (~18.8 per video on average), which is
close to the expected 360. The remaining shortfall is explained by the OHP bad form videos
where the no-lockout fault produces a flatter elbow angle signal that is harder to segment.

---

### Subject-Dependent Evaluation (Data Leakage)
**Discovered:** During initial model training (Phase 3)

**Problem:**
The initial train/test split was performed randomly across all repetitions using an 80/20
split (`train_test_split`). Since all videos were recorded by the same person in the same
session, repetitions from the same video could appear in both the training set and the test
set. This means the model was evaluated on data that is nearly identical to what it trained
on — same person, same camera angle, same movement style.

The result was 100% accuracy on all three exercises, which is not a credible reflection of
real-world generalization.

**Root cause:**
This is a form of data leakage known as subject-dependent evaluation. The model learns the
specific movement signature of one individual rather than a generalizable notion of correct
vs incorrect form.

**Fix:**
Replaced random split with leave-one-video-out (LOVO) cross-validation. For each exercise
(6 videos), the model is trained on 5 videos and tested on the held-out 6th, rotating
through all 6 combinations. This ensures the test set never contains reps from a video
that was used during training, producing a more honest accuracy estimate.

**Remaining limitation:**
Even with LOVO, all data comes from a single subject. The model will not generalize to
other people without retraining on their data. This is a known limitation of small,
single-subject datasets and should be acknowledged in any reported results.

---

### LOVO Evaluation Results
**Squat and Deadlift: 100% accuracy**
The angle differences between good and bad form are large enough that the classifier
separates them perfectly even across different videos of the same person. For squats, the
knee angle minimum averages 36.5° for good form vs 131.6° for bad form — a difference
large enough that there is essentially no overlap between classes.

**OHP: 99% accuracy (1 misclassification)**
One bad form rep was classified as good. This is the most realistic result since the
no-lockout fault (elbow_angle_max: 173° good vs 163° bad) is a subtler difference than
squat depth. The single error confirms the evaluation is working honestly — a perfect
score on all three after fixing the leakage issue would have been suspicious.

**Conclusion:**
The high accuracy is credible given the large biomechanical differences between form
classes and the consistent recording conditions. However, results should be interpreted
as single-subject performance only. Generalization to other individuals requires
additional data collection and retraining.