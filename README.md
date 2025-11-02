# 🧩 Jumbled Video Frames Reconstruction  
### Internship Task — Tecdia

---

## 🎯 Project Overview

This project reconstructs a **jumbled or shuffled video** by automatically detecting the correct order of its frames.  
It uses a **frame similarity–based algorithm** that analyzes visual similarity between frames to restore the original smooth sequence.

---

## ⚙️ Core Concept

In a normal video, consecutive frames are visually similar because they capture small motion changes.  
When frames are shuffled, that continuity breaks.  
This algorithm computes **frame-to-frame similarity** and arranges frames so that the overall visual change is minimal — recreating the correct playback order.

---

## 🧠 Algorithm Used

### 1. Frame Similarity — Mean Squared Error (MSE)
Each frame is converted to grayscale and compared with all others using:
\[
MSE(A, B) = \frac{1}{N} \sum_{i=1}^{N} (A_i - B_i)^2
\]
Lower MSE = higher similarity.

### 2. Starting Frame Selection
The frame with the **highest average MSE** (least similar to others) is assumed to be the **starting frame**, since start or end frames often have unique content.

### 3. Greedy Reconstruction
From the start frame, the algorithm repeatedly picks the **most similar next frame** (lowest MSE) to form a likely forward sequence.

### 4. Local Smoothing
Small local swaps are performed to correct minor ordering mistakes, improving smoothness and continuity.

### 5. Direction Auto-Detection
Both forward and reversed sequences are evaluated.  
The direction with **lower total discontinuity** (sum of differences between consecutive frames) is selected automatically.

---

## 🧩 Algorithm Summary Table

| Step | Technique | Purpose |
|------|------------|----------|
| 1 | Frame Extraction (OpenCV) | Read video frames |
| 2 | Grayscale Conversion | Simplify similarity calculation |
| 3 | MSE Computation | Build pairwise difference matrix |
| 4 | Start Frame Detection | Identify likely first frame |
| 5 | Greedy Ordering | Construct frame sequence |
| 6 | Local Smoothing | Fix local misplacements |
| 7 | Direction Detection | Auto-select correct playback direction |
| 8 | Video Writer | Save reconstructed video |

---

## 🧰 Requirements

Install the required Python packages before running:

```bash
pip install -r requirements.txt
