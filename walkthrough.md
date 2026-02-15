# Sono-Guide Prototype Walkthrough

We have successfully implemented the core logic for the Sono-Guide hackathon prototype.

## Features Implemented

### 1. Calibration Mode (Pre-scan)
The system now analyzes the incoming video feed for quality issues:
*   **Too Dark:** Checks if mean pixel intensity < 40.
*   **High Noise:** Checks if standard deviation > 80 (simulated graininess check).
*   **Out of Focus:** Checks if Laplacian variance < 100.
*   **UI Feedback:** The Status Label turns RED and lists specific issues if any are found.

### 2. AI & Standard Plane Simulation
We integrated `ultralytics` (YOLOv8) to simulate the "Fetal Head" detection:
*   **Detection:** Uses YOLOv8n to detect objects (e.g., person, cell phone, cup) as a proxy for the fetal head.
*   **Standard Plane Logic:**
    *   **Condition 1:** High Confidence Detection (> 0.6).
    *   **Condition 2:** Object is **Centered** in the frame (within the middle 15% margin).
    *   **Condition 3:** Image Calibration is OK.
*   **Visuals:**
    *   **Red Box:** Object detected but not optional.
    *   **Green Box:** Object is in the "Standard Plane".

### 3. Auto-Freeze Functionality
*   **Stability Counter:** If the Green Box is maintained for ~2 seconds (30 frames), the system automatically captures the frame.
*   **Feedback:** Saves the image to disk as `snapshot_{timestamp}.jpg` and flashes a confirmation message.

## How to Test

1.  **Prepare the Feed:**
    *   Ensure you have a video file named `simulation_feed.mp4` in the project directory.
    *   If testing the "Standard Plane", try to find a video where a distinct object passes through the center.
    *   *Hackathon Tip:* If you don't have a fetal ultrasound video, record a quick video of yourself moving a coffee cup into the center of the frame. It works perfect for the demo!

2.  **Run the App:**
    ```bash
    python main.py
    ```
    *   (First run will download the YOLO model, ~6MB).

3.  **Verify Calibration:**
    *   Cover the camera (if using webcam source) or use a dark video to test the "Too Dark" warning.

4.  **Verify AI:**
    *   Watch for the Red Box detecting objects.
    *   Wait for the object to center.
    *   Watch the box turn Green.
    *   Wait 2 seconds -> Check the folder for the saved snapshot.
