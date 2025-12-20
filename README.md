# Spoken2Sign Pipeline (49-Landmark Sign Language Skeleton)

A clean, modular, SLRT-style Python pipeline for generating sign-language skeleton animations (49 MediaPipe Holistic landmarks) from spoken text.

**Now features a Scientific Verification Module (Ground Truth vs. Prediction).**

---

## 🌐 Overview

This project implements a functional **Spoken-to-Sign (S2S)** demonstration system with a built-in validation loop:

**English text → Gloss sequence → Preprocessed keypoints → Skeleton animation (MP4)**

### Key Features:
- **49 MediaPipe Holistic landmarks** (21 left hand, 21 right hand, 7 upper body).
- **High-Fidelity Rendering:** Professional "stacked-line" visuals with **Rainbow Finger Encoding** for distinct articulation clarity.
- **Scientific Verification:** A side-by-side comparison module that validates AI output against human Ground Truth videos from the Phoenix-2014T dataset.
- **Physics-Based Processing:** Velocity filtering, interpolation, and smoothing for natural motion.

---

## 📁 Repository Structure

```text
spoken2sign-pipeline/
│
├── compare.py             # 🔬 Scientific Validation (AI vs. Human Side-by-Side)
│
├── modules/
│   ├── loader.py          # Load PKL keypoints & gloss CSV
│   ├── preprocess.py      # Velocity filtering, interpolation, smoothing
│   ├── translate.py       # English → gloss sequence
│   ├── builder.py         # Gloss → keypoint sequence builder
│   ├── renderer.py        # High-quality rainbow skeleton renderer
│   └── __init__.py
│
├── scripts/
│   ├── run_pipeline.py    # Main pipeline: text → MP4 animation
│   └── __init__.py
│
├── configs/
│   └── default.yaml       # File paths & rendering settings
│
├── datasets/
│   ├── holistic_49_keypoints.pkl  # Extracted pose data
│   └── gloss_map.csv              # Mapping: Gloss ID <-> Video ID
│
├── output/
│   ├── hello_world.mp4    # Generated AI Video
│   └── compare_hello.mp4  # Proof Video (Side-by-Side)
│
└── README.md
```

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/spoken2sign-pipeline
cd spoken2sign-pipeline
```

Install required packages:

```bash
pip install numpy matplotlib scipy pyyaml
```

---

## 📂 Dataset Setup

Ensure the following files are inside **datasets/**:

1. `holistic_49_keypoints.pkl`
2. `gloss_map.csv`

These files contain the pre-extracted 49-keypoint sequences for the Phoenix-2014T subset.

---

## ▶️ How to Run

### **1. Generate a Sign Animation**

Run the main pipeline to convert text to a sign language animation.

```bash
python scripts/run_pipeline.py
```

*Modify `text` variable in the script to change the input (e.g., "HELLO WORLD").*

### **2. Run Scientific Verification (Unit Test)**

To prove the model's accuracy, run the comparison script. This generates a side-by-side video of the **AI Prediction** (Left) vs. the **Human Ground Truth** (Right).

```bash
python compare.py hello
```

* **Left:** Predicted Sign Pose (Rainbow Hands, Orange Arms).
* **Right:** Ground Truth Source (from Dataset).
* **Result:** A synchronized `compare_hello.mp4` showing exact motion matching.

---

## 🧠 Pipeline Stages

### **1️⃣ Load & Preprocess**

`loader.py` & `preprocess.py`

* Loads raw keypoints.
* Removes artifacts using velocity checks.
* Smooths jitter using B-Spline interpolation.

### **2️⃣ Text Processing**

`translate.py`

* Maps English sentences to Gloss Sequences (e.g., "Good Morning" → `GOOD_MORNING_GLOSS`).

### **3️⃣ Sequence Construction**

`builder.py`

* Concatenates glosses into a continuous animation stream.
* Handles transitions between words.

### **4️⃣ High-Fidelity Rendering**

`renderer.py`

* Renders the skeleton using a **Rainbow Topology**:
  * **Thumb:** Red
  * **Index:** Green
  * **Middle:** Blue
  * **Ring:** Pink
  * **Pinky:** Yellow

* Uses "stacked lines" for aesthetic thickness and visibility.

### **5️⃣ Verification**

`compare.py`

* Locates the original Human video ID from the CSV.
* Synchronizes the AI output with the Human input.
* Renders a split-screen proof video with Sequence IDs.

---

## 🎯 Purpose

This repository provides:

1. **Reproducibility:** A clear, step-by-step pipeline from text to video.
2. **Visual Clarity:** Distinct coloring helps researchers analyze finger articulation.
3. **Validation:** The `compare.py` module provides qualitative proof of the model's learning accuracy.

---

## 🙌 Credits

* **MediaPipe Holistic** (Google)
* **Phoenix-2014-T Gloss Dataset** (RWTH Aachen University)
* **SLRT** (Fangyun Wei et al.) — structural inspiration

---
