# Exercise Form Classifier – Final Project Presentation Script

---

## Slide 1 – Title Slide

### Script

Today I’ll be presenting my final project, an exercise form classification system built using computer vision and machine learning.

The goal of this project was to analyze exercise form using only smartphone video, without requiring wearable sensors or specialized equipment.

The system focuses on three exercises:

* squat,
* overhead press,
* and deadlift.

The pipeline takes a video, extracts body landmarks, converts them into joint angles, segments repetitions, and classifies whether the form is correct or incorrect.

### Transition

First, I’ll explain the motivation behind the project.

---

## Slide 2 – Motivation

### Script

Incorrect exercise form is one of the main causes of lifting-related injuries.

Most people rely on coaches for feedback, but coaching is not always accessible or affordable.

The goal of this project was to explore whether pose estimation and machine learning could provide a low-cost alternative using only a phone camera.

### Transition

I’ll now walk through how the system was built.

---

## Slide 3 – Data Collection

### Script

I recorded all videos myself using an iPhone 15 Pro Max at 720p and 30 frames per second.

All recordings were captured from a side angle because it makes movement mechanics easier to analyze.

For each exercise, I recorded both good-form and bad-form examples.

Examples of incorrect form included:

* insufficient squat depth,
* incomplete lockout during overhead press,
* and lumbar rounding during deadlifts.

### Transition

After collecting the videos, the next step was pose estimation.

---

## Slide 4 – Pose Estimation

### Script

For pose estimation, I used Google’s MediaPipe BlazePose framework.

The model outputs 33 body landmarks, but I only used the joints most relevant to the exercises, such as the shoulders, hips, knees, and ankles.

Frames with low-confidence landmark detection were filtered out during preprocessing to improve reliability.

### Transition

Once the landmarks were extracted, I converted them into biomechanical measurements.

---

## Slide 5 – Joint Angle Calculation

### Script

The raw landmark coordinates were converted into joint angles using vector-based angle calculations.

For example:

* knee angle measured squat depth,
* elbow angle evaluated overhead press lockout,
* and trunk angle detected excessive forward lean.

Using angles made the system more interpretable and computationally efficient.

### Transition

Next, the system needed to identify individual repetitions.

---

## Slide 6 – Repetition Segmentation

### Script

To isolate repetitions, I used peak and valley detection on the joint-angle signals.

Different exercises used different segmentation signals:

* squats used knee-angle valleys,
* overhead press used elbow-angle peaks,
* and deadlifts used hip-angle peaks.

The signals were smoothed beforehand to reduce pose-estimation noise.

### Transition

After segmentation, I moved into the machine learning stage.

---

## Slide 7 – Classification

### Script

For each repetition, the system calculated statistical features such as:

* minimum angle,
* maximum angle,
* mean,
* and range of motion.

These features were used to train a Random Forest classifier.

I selected Random Forest because it performs well on smaller datasets and is computationally efficient.

Separate classifiers were trained for each exercise type.

### Transition

I’ll now briefly discuss the results.

---

## Slide 8 – Results

### Script

Overall, the project demonstrated that pose-estimation-based movement analysis can distinguish between correct and incorrect exercise form using only smartphone video.

The complete pipeline successfully handled:

* pose extraction,
* joint-angle computation,
* repetition segmentation,
* and final classification.

The project also showed that biomechanical features can be effective even with a relatively small dataset.

### Transition

There are still several areas for future improvement.

---

## Slide 9 – Future Work

### Script

One limitation of the current system is the relatively small dataset and fixed camera angle.

Future work could include:

* larger and more diverse datasets,
* additional exercises,
* real-time mobile feedback,
* and deep learning models such as LSTMs or Transformers.

The system could also evolve into a coaching assistant capable of giving corrective feedback instead of only classifying form quality.

### Transition

Finally, I’ll demonstrate the system using sample exercise footage.

---

## Slide 10 – Demo

### Script

For the demo, I’ll show:

* pose tracking,
* joint-angle extraction,
* repetition segmentation,
* and the final classification output.

This demonstrates the complete workflow from raw video input to prediction.

---

## Closing Statement

### Script

In summary, this project explored how computer vision and machine learning can be combined to create a low-cost exercise form classification system using only smartphone video.

Thank you for your time.
