# Spoken2Sign Pipeline (49-Landmark Sign Language Skeleton)

A clean, modular, SLRT-style Python pipeline for generating sign-language skeleton animations (49 MediaPipe Holistic landmarks) from spoken text.

**Now features a Scientific Verification Module (Ground Truth Comparison & Quantitative Metrics).**

---

## 🌐 Overview

This project implements a functional **Spoken-to-Sign (S2S)** demonstration system with a built-in validation loop:

**English text → Gloss sequence → Preprocessed keypoints → Skeleton animation (MP4)**

### Key Features:

* **49 MediaPipe Holistic landmarks** (21 left hand, 21 right hand, 7 upper body).
* **High-Fidelity Rendering:** Professional "stacked-line" visuals with **Rainbow Finger Encoding** for distinct articulation clarity.
* **Scientific Verification:** A side-by-side comparison module that validates AI output against human Ground Truth videos.
* **Quantitative Metrics:** Built-in evaluation of Geometric Accuracy (MPJPE) and Visual Fidelity (FID).

---

## 📁 Repository Structure

```text
spoken2sign-pipeline/
│
├── compare.py             # 🔬 Scientific Validation (AI vs. Human Side-by-Side)
├── evaluate.py            # 📊 Quantitative Metrics (MPJPE, MPJAE, DTW, FID)
├── render_gt.py           # 🎬 Ground Truth Renderer (High-Fidelity)
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
pip install numpy matplotlib scipy pyyaml torch torchvision opencv-python

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

### **2. Generate Ground Truth (Visual Twin)**

Render the original human motion with the exact same visual style as the AI output.

```bash
python render_gt.py hello_world

```

### **3. Run Quantitative Evaluation (Metrics)**

Calculate scientific accuracy scores (MPJPE, DTW, FID).

```bash
python evaluate.py hello_world

```

**Output Example:**

* **MPJPE (Position Error):** `0.5368` (Geometric Accuracy)
* **MPJAE (Angle Error):** `20.38°` (Articulation Correctness)
* **DTW (Time Warping):** `2.33` (Temporal Alignment)
* **FID (Visual Quality):** `122.66` (Generative Fidelity)

### **4. Visual Comparison (Side-by-Side)**

Generate a split-screen video to visually verify the result.

```bash
# Uses FFmpeg to stitch videos
ffmpeg -i output/hello_world.mp4 -i output/hello_world_gt.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" output/final_comparison.mp4

```

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

---

## 🎯 Purpose

This repository provides:

1. **Reproducibility:** A clear, step-by-step pipeline from text to video.
2. **Visual Clarity:** Distinct coloring helps researchers analyze finger articulation.
3. **Validation:** The `evaluate.py` module provides research-grade metrics to prove model accuracy.

---

## 🙌 Credits

* **MediaPipe Holistic** (Google)
* **Phoenix-2014-T Gloss Dataset** (RWTH Aachen University)
* **SLRT** (Fangyun Wei et al.) — structural inspiration

---
