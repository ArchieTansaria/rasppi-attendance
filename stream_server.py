import cv2
import time
from flask import Flask, Response
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.face_recognizer import FaceRecognizer
from src.attendance import AttendanceManager
from src.utils import ensure_directories
from collections import deque, Counter

app = Flask(__name__)

# Initialize components
ensure_directories()
recognizer = FaceRecognizer()
attendance_manager = AttendanceManager()

# For Temporal Smoothing: name_history for EACH person detected
# In a real multi-person system, you'd track face IDs, 
# but for simple 1-2 person logic, we'll keep a window of recent names.
identity_history = deque(maxlen=config.TEMPORAL_WINDOW)

def get_smoothed_name(detected_name):
    """Adds a detection to history and returns the most frequent stable name."""
    identity_history.append(detected_name)
    
    # Count occurrences
    counts = Counter(identity_history)
    most_common, count = counts.most_common(1)[0]
    
    # Stability threshold: Must be seen in majority of frames
    if count >= (config.TEMPORAL_WINDOW // 2 + 1):
        return most_common
    return "Searching..."

def generate_frames():
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    frame_count = 0
    recognized_faces = []

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        # Recognition logic (every Nth frame)
        if frame_count % config.FRAME_SKIP == 0:
            raw_faces = recognizer.recognize_faces(frame)
            
            # Apply Temporal Smoothing to the main face
            recognized_faces = []
            for name, bbox in raw_faces:
                stable_name = get_smoothed_name(name)
                recognized_faces.append((stable_name, bbox))
                
                # Log attendance only if the name is STABLE and not Unknown/Searching
                if stable_name not in ["Unknown", "Searching..."]:
                    attendance_manager.mark_attendance(stable_name)
        
        frame_count += 1

        # Draw boxes on the frame for the stream
        for name, (top, right, bottom, left) in recognized_faces:
            # Color: Green for stable match, Yellow for searching, Red for unknown
            if name == "Searching...":
                color = (0, 255, 255) # Yellow
                text_color = (0, 0, 0)
            elif name == "Unknown":
                color = (0, 0, 255) # Red
                text_color = (255, 255, 255)
            else:
                color = (0, 255, 0) # Green
                text_color = (0, 0, 0)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, text_color, 1)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page."""
    return """
    <html>
      <head>
        <title>AI Attendance Stream</title>
      </head>
      <body>
        <h1>AI Attendance Stream Server</h1>
        <p>The server is running correctly.</p>
        <ul>
          <li><b>Video Feed:</b> <a href="/video">/video</a></li>
        </ul>
        <hr>
        <p>Use this URL in your Streamlit dashboard: <code>http://&lt;your-pi-ip&gt;:8000/video</code></p>
      </body>
    </html>
    """

@app.route('/video')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"[INFO] Starting Stream Server on {config.STREAM_HOST}:{config.STREAM_PORT}...")
    print(f"[INFO] Access the stream at http://<pi-ip>:{config.STREAM_PORT}/video")
    app.run(host=config.STREAM_HOST, port=config.STREAM_PORT, threaded=True)
