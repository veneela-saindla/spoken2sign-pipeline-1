# Spoken2Sign Pipeline (49-Landmark Sign Language Skeleton)

A clean, modular, SLRT-style Python pipeline for generating sign-language skeleton animations (49 MediaPipe Holistic landmarks) from spoken text.

---

## 🌐 Overview

This project implements a functional **Spoken-to-Sign (S2S)** demonstration system:

**English text → Gloss sequence → Preprocessed keypoints → Skeleton animation (MP4)**

It uses:

- **49 MediaPipe Holistic landmarks**  
  - 21 left hand  
  - 21 right hand  
  - 7 upper body  
- Phoenix-style gloss mapping (GLOSS_0 – GLOSS_19)  
- Clean py modules & scripts  
- Interpolation, velocity filtering, and smoothing  
- A layered renderer for human-readable sign skeletons  

This structure makes the system extensible for future training, evaluation, or integration with SLRT-style models.

---

## 📁 Repository Structure

```
spoken2sign-pipeline/
│
├── modules/
│   ├── loader.py          # Load PKL keypoints & gloss CSV
│   ├── preprocess.py      # Velocity filtering, interpolation, smoothing
│   ├── translate.py       # English → gloss sequence
│   ├── builder.py         # Gloss → keypoint sequence builder
│   ├── renderer.py        # Layered skeleton renderer (MP4 output)
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
│   └── README.md          # Instructions to place PKL & CSV files
│
├── output/
│   └── .gitkeep           # Rendered videos saved here
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

Install required packages (Colab already has them):

```bash
pip install numpy matplotlib scipy pyyaml
```

---

## 📂 Dataset Setup

Place the following files inside **datasets/**:

- `holistic_49_keypoints.pkl`
- `gloss_map.csv`

These files contain the pre-extracted 49-keypoint sequences for each gloss (Phoenix-small subset).

Then, edit the paths inside:

```
configs/default.yaml
```

---

## ▶️ Run the Pipeline

From the project root:

```bash
python scripts/run_pipeline.py
```

Videos will be saved in:

```
output/
```

The pipeline generates animations for test sentences such as:

- HELLO WORLD  
- GOOD MORNING  
- THANK YOU  
- WHAT IS YOUR NAME  
- STOP PLEASE  
- HAPPY MORNING  

---

## 🧠 Pipeline Stages

### **1️⃣ Load Data**
`loader.py`

Loads PKL keypoints + gloss map.

---

### **2️⃣ Preprocess Sequence**
`clean_sequence()` in `preprocess.py`

✔ Removes unrealistic jumps  
✔ Interpolates missing frames  
✔ Smooths hands and upper-body joints  

---

### **3️⃣ Text → Gloss Mapping**
`translate.py`

Maps English words to gloss IDs.

---

### **4️⃣ Build Sequence**
`builder.py`

Creates a continuous sequence with transitions between glosses.

---

### **5️⃣ Render Animation**
`renderer.py`

Outputs clean, layered skeleton animations:

- Grey body  
- Red hands  
- Blue joints  

---

## 🎯 Purpose

This repository provides:

- A reproducible Spoken-to-Sign demonstration pipeline  
- Proper research-grade project structuring  
- A clean transition away from notebooks to py scripts  
- A foundation for integrating SLRT-style models in future work  

---

## 🙌 Credits

- MediaPipe Holistic  
- Phoenix-2014-T Gloss Dataset  
- SLRT (Fangyun Wei et al.) — for structural inspiration  

---
