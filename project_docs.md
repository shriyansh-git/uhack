# Sono-Guide Project Plan

## 1. The Stack Decision
For a 24-hour hackathon, we prioritize **ease of use, speed of implementation, and community support**.

*   **Language:** Python 3.10+
*   **Core Video Processing:** `opencv-python` (cv2)
    *   *Why:* Industry standard, handles video file looping and pixel manipulation (for calibration) effortlessly.
*   **AI/Inference:** `ultralytics` (YOLOv8)
    *   *Why:* Extremely fast to implement (3 lines of code), runs decent on CPU if needed, and has pre-trained models we can fine-tune or use generically if we just need to detect "people" or specific shapes as a proxy.
*   **GUI:** `tkinter` (with `Pillow` for image display)
    *   *Why:* built-in, lightweight, and allows easy overlay of text/buttons without the complexity of a web server (saving overhead vs Streamlit/Flask for a real-time 30fps video feed).
*   **Data/Arrays:** `numpy`
    *   *Why:* Required for OpenCV image math.

### Recommended `requirements.txt`:
```text
opencv-python
ultralytics
Pillow
numpy
```

---

## 2. The 24-Hour Sprint Plan

### Block 1: The Foundation (Hours 0-4)
*   **Goal:** A running App window playing the video loop.
*   **Tasks:**
    *   Set up git repo & virtual env.
    *   Build `main.py` scaffold (Tkinter window + OpenCV loop).
    *   Implement "Simulation Mode": Load a standard MP4 file and loop it endlessly.
    *   Ensure >15 FPS playback in the GUI.

### Block 2: Calibration Logic (Hours 4-8)
*   **Goal:** System can complain if the image is bad.
*   **Tasks:**
    *   Implement `assess_image_quality(frame)` function.
    *   Add check for **Brightness**: Mean pixel intensity < Threshold (Too Dark).
    *   Add check for **Noise**: Variance/Gain check (Too Grainy).
    *   Add UI Overlay: "Please Adjust Gain" warning message.

### Block 3: AI Model Setup (Hours 8-12)
*   **Goal:** The model "sees" something.
*   **Tasks:**
    *   Install YOLOv8.
    *   (Optional) If you have labeled data, start a quick training run on Colab.
    *   If no data, use a pre-trained model to detect *anything* (like a "cup" or "cell phone") as a proxy for the "Fetal Head" to test the pipeline.
    *   Draw the raw bounding boxes on the video feed.

### Block 4: Logic & "Standard Plane" (Hours 12-16)
*   **Goal:** The Logic Layer (Red Box -> Green Box).
*   **Tasks:**
    *   Implement state tracking: Is the detected object centered? Is the confidence score high (> 0.8)?
    *   Create the "Stability Counter": Needs to stay good for 2 consecutive seconds (e.g., 60 frames).
    *   Visual Feedback: Change box color from Red to Green dynamically.

### Block 5: Auto-Freeze & Interaction (Hours 16-20)
*   **Goal:** The "Money Shot" feature.
*   **Tasks:**
    *   Implement `save_frame(frame)` execution when Stability Counter hits target.
    *   Update UI to show a "Snapshot Captured" notification.
    *   Add "Freeze" button to manual override.

### Block 6: Polish & Demo Prep (Hours 20-24)
*   **Goal:** Make it look like a medical device.
*   **Tasks:**
    *   Style the UI (Dark mode, professional fonts).
    *   Record the demo video (or prep the live demo script).
    *   Code cleanup/refactoring.
