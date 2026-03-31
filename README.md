# 🧍‍♂️ ErgoTool: Real-Time Ergonomic Posture Risk Assessment

**ErgoTool** is a computer vision-based application designed to analyze human posture in real-time and assess ergonomic risks. Using a standard webcam, the tool tracks body landmarks, calculates joint angles and bodily alignment, and logs risk scores to help prevent musculoskeletal strain.

## 📖 Overview
Prolonged poor posture can lead to severe health issues. ErgoTool automates the evaluation process by applying mathematical heuristics to body tracking data. It categorizes posture into **Low**, **Medium**, or **High Risk** and logs the data for long-term analysis.

## 🧠 How It Works (The Logic)
The core of ErgoTool relies on **Google's MediaPipe Pose** for real-time 3D landmark detection and **OpenCV** for image processing. 

Once the body landmarks are identified, the script evaluates three critical ergonomic factors:

1. **Shoulder & Hip Imbalance:** The script checks the vertical ($y$-axis) difference between the left and right shoulders, as well as the left and right hips. If the difference exceeds a threshold of `0.1` (indicating leaning or slouching to one side), a risk point is added.
2. **Spine Alignment:** It calculates the midpoint between the shoulders and the midpoint between the hips. If the vertical deviation between these two midpoints exceeds `0.2`, it indicates a bent or misaligned spine, adding another risk point.
3. **Knee Angles:** Using inverse trigonometric functions (`arctan2`), the tool calculates the exact angle formed by the hip, knee, and ankle. If either knee is bent at an angle tighter than `160 degrees`, a risk point is added.

### 📊 The Scoring System
The accumulated risk points translate into a clear, actionable score:
* **0 Points:** `Low Risk` (Healthy posture)
* **1 Point:** `Medium Risk` (Minor postural deviation)
* **2 or 3 Points:** `High Risk` (Poor posture requiring immediate correction)

## 🛠️ Prerequisites
To run ErgoTool, you will need **Python 3.7+** and a working webcam. 

Install the required Python libraries using `pip`:

```bash
pip install opencv-python mediapipe numpy pandas
