import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import PIL.Image, PIL. ImageTk
import numpy as np
import os
from ultralytics import YOLO
import time

# --- Configuration ---
VIDEO_SOURCE = "simulation_feed.mp4"  # Path to your recorded ultrasound video
Simulate_Hardware = True

class SonoGuideApp:
    def __init__(self, window, video_source=VIDEO_SOURCE):
        self.window = window
        self.window.title("Sono-Guide: AI Ultrasound Co-Pilot")
        self.window.geometry("1024x768")
        self.window.configure(bg="#1e1e1e") # Dark mode background

        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)

        # --- UI Layout ---
        self._create_header()
        self._create_video_display()
        self._create_controls()

        # --- AI Configuration ---
        # Using yolov8n.pt (Nano) for speed. It will download automatically on first run. 
        # For the hackathon, we will detect 'person' or 'cell phone' as a proxy for 'fetal head'.
        print("Loading AI Model...")
        self.model = YOLO("yolov8n.pt") 
        print("AI Model Loaded.")
        
        # --- State Variables ---
        self.calibration_ok = False
        self.ai_active = True
        self.frame_count = 0
        self.stable_frames = 0 # Counter for auto-freeze
        self.last_saved_time = 0 
        
        # Start the update loop
        self.delay = 15 # ms
        self.update()

    def _create_header(self):
        header_frame = tk.Frame(self.window, bg="#2d2d2d", height=50)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        
        title_label = tk.Label(header_frame, text="Sono-Guide Prototype", 
                                font=("Helvetica", 18, "bold"), 
                               bg="#2d2d2d", fg="#ffffff")
        title_label.pack(pady=10)

    def _create_video_display(self):
        # Frame to hold the canvas
        self.video_frame = tk.Frame(self.window, bg="#000000")
        self.video_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Canvas for video
        self.canvas = tk.Canvas(self.video_frame, bg="#000000", width=800, height=600)
        self.canvas.pack(anchor=tk.CENTER, expand=True)

    def _create_controls(self):
        control_panel = tk.Frame(self.window, bg="#2d2d2d", height=100)
        control_panel.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status Label
        self.status_label = tk.Label(control_panel, text="System Status: INITIALIZING...", 
                                     font=("Consolas", 12), bg="#2d2d2d", fg="#00ff00")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Buttons (Placeholders)
        btn_toggle = tk.Button(control_panel, text="Toggle AI Overlay", command=self.toggle_ai,
                               bg="#444444", fg="white", font=("Arial", 10))
        btn_toggle.pack(side=tk.RIGHT, padx=20, pady=20)

    def toggle_ai(self):
        self.ai_active = not self.ai_active
        status = "ON" if self.ai_active else "OFF"
        print(f"AI Overlay: {status}")

    def update(self):
        # 1. Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            # 2. Run Calibration Checks (Placeholder)
            self.run_calibration_logic(frame)

            # 3. Run AI Inference (Placeholder)
            if self.ai_active:
                frame = self.run_ai_logic(frame)

            # 4. Display frame
            # Convert OpenCV image (BGR) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create PIL Image
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_rgb))
            # Update Canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            # Resize canvas to fit frame if needed (optional optimization)
            
        
        self.frame_count += 1
        self.window.after(self.delay, self.update)

    def run_calibration_logic(self, frame):
        """
        Runs calibration checks on the current frame.
        - Brightness: Mean pixel intensity.
        - Noise (Graininess): Standard deviation of pixel intensities.
        - Focus: Variance of Laplacian.
        """
        # Run every 5 frames to save processing power
        if self.frame_count % 5 != 0:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Brightness Check
        mean_brightness = np.mean(gray)
        
        # 2. Noise/Gain Check (Simplified: High std dev often means high contrast/noise)
        # A more specific noise check might involve high frequency analysis, 
        # but std_dev is a decent proxy for "too much activity/gain" in a dark context.
        std_dev = np.std(gray)
        
        # 3. Focus Check (Variance of Laplacian)
        # Low variance = blurry. High variance = sharp edges.
        focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()

        # --- Thresholds (Tune these based on actual video) ---
        BRIGHTNESS_LOW_THRESH = 40.0
        NOISE_HIGH_THRESH = 80.0 # Arbitrary start point for "Too Grainy"
        FOCUS_LOW_THRESH = 100.0 # Blurry if below this

        status_text = "Status: CALIBRATION OK"
        status_color = "#00ff00" # Green

        issues = []
        if mean_brightness < BRIGHTNESS_LOW_THRESH:
            issues.append("TOO DARK (Increase Gain)")
        
        if std_dev > NOISE_HIGH_THRESH:
             # This can be tricky; high contrast is good, but too much 'static' is bad.
             # We might label this as "HIGH GAIN" for now.
             issues.append("HIGH NOISE (Lower Gain)")

        if focus_measure < FOCUS_LOW_THRESH:
            issues.append("OUT OF FOCUS")

        if issues:
            status_text = "WARNING: " + ", ".join(issues)
            status_color = "red"
            self.calibration_ok = False
        else:
            self.calibration_ok = True

        # Update UI (safely)
        self.status_label.config(text=status_text, fg=status_color)

    def run_ai_logic(self, frame):
        """
        Runs YOLOv8 inference and handles the 'Standard Plane' logic.
        """
        # 1. Run Inference
        results = self.model(frame, verbose=False)
        
        # 2. Parse Detections
        # We'll take the highest confidence detection as our target
        target_box = None
        max_conf = 0.0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Class 0 is 'person' in COCO. 
                # If testing with a phone video, maybe check for cell phone (class 67).
                # For this generic demo, we'll take *any* high-conf detection.
                conf = float(box.conf[0])
                if conf > 0.4 and conf > max_conf:
                    max_conf = conf
                    target_box = box

        height, width, _ = frame.shape
        
        if target_box is not None:
            # Unpack coordinates
            x1, y1, x2, y2 = map(int, target_box.xyxy[0])
            w_box, h_box = x2-x1, y2-y1
            cx, cy = x1 + w_box//2, y1 + h_box//2 # Center of box

            # 3. Check for "Standard Plane" (Simulated)
            # Criteria: 
            #   - Confidence > 0.6
            #   - Object Centered (within middle 30% of screen)
            #   - Calibration is OK
            
            center_margin_x = width * 0.15
            center_margin_y = height * 0.15
            is_centered = (width/2 - center_margin_x < cx < width/2 + center_margin_x) and \
                          (height/2 - center_margin_y < cy < height/2 + center_margin_y)
            
            is_standard_plane = (max_conf > 0.6) and is_centered and self.calibration_ok

            # 4. Draw Overlay
            color = (0, 0, 255) # Red (Default)
            label = f"Searching... {max_conf:.2f}"
            thickness = 2
            
            if is_standard_plane:
                self.stable_frames += 1
                color = (0, 255, 0) # Green
                label = f"OPTIMAL VIEW ({self.stable_frames}/30)"
                thickness = 4
                
                # Progress Bar / Visual Indicator of Stability
                # (Simple overlay text for now)
                
                # 5. Check Threshold for Auto-Freeze
                # Assuming ~15 FPS in Tkinter loop, 2 seconds = 30 frames
                if self.stable_frames >= 30:
                    self.perform_auto_freeze(frame)
                    self.stable_frames = 0 # Reset or pause
            else:
                self.stable_frames = 0 # Reset counter if conditions broken

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw Center Target Support Lines (to help user aim)
            cv2.circle(frame, (int(width/2), int(height/2)), 5, (255, 255, 0), -1)

        else:
             self.stable_frames = 0

        return frame

    def perform_auto_freeze(self, frame):
        current_time = time.time()
        # Prevent double-saves (cooldown 3 seconds)
        if current_time - self.last_saved_time < 3.0:
            return

        print("AUTO-FREEZE TRIGGERED!")
        
        # Save Image
        filename = f"snapshot_{int(current_time)}.jpg"
        cv2.imwrite(filename, frame)
        self.last_saved_time = current_time
        
        # Flash UI Feedback
        self.status_label.config(text=f"SNAPSHOT SAVED: {filename}", fg="#00ffff")


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        try:
            self.vid = cv2.VideoCapture(video_source)
            if not self.vid.isOpened():
                raise ValueError("Unable to open video source", video_source)
            
            # Get video source width and height
            self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Video Source Loaded: {self.width}x{self.height}")
            
        except ValueError:
            print(f"Video file '{video_source}' not found. Using simulation noise.")
            self.vid = None
            self.width = 640
            self.height = 480

    def get_frame(self):
        if self.vid and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, frame)
            else:
                # Loop video: Set frame position back to 0
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return self.vid.read()
        else:
            # Generate static noise if no video found
            noise = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            return (True, noise)

    def __del__(self):
        if self.vid:
            self.vid.release()

if __name__ == "__main__":
    # Create the App
    root = tk.Tk()
    app = SonoGuideApp(root, VIDEO_SOURCE)
    root.mainloop()
