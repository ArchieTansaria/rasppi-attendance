import cv2
import os
import time
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.utils import ensure_directories

def capture_samples(student_name):
    """
    Captures 5-10 images of a student using the current camera (e.g., Pi Digicam).
    Ensures that training quality matches testing quality.
    """
    ensure_directories()
    student_dir = os.path.join(config.STUDENTS_DIR, student_name)
    os.makedirs(student_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    print(f"\n--- CAPTURING SAMPLES FOR {student_name.upper()} ---")
    print("Instructions:")
    print("1. Look directly at the camera.")
    print("2. Tilt your head slightly (left, right, up, down).")
    print("3. We will capture 10 photos. Ready?")
    time.sleep(2)

    count = 0
    while count < 10:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Display feedback (can't use imshow on headless Pi, so we'll just save)
        # On a headless Pi, we just rely on the user being in front of the camera.
        
        timestamp = int(time.time() * 1000)
        img_path = os.path.join(student_dir, f"{student_name}_{count}_{timestamp}.jpg")
        cv2.imwrite(img_path, frame)
        
        print(f"[SAVED] {count+1}/10: {img_path}")
        count += 1
        time.sleep(1) # Wait 1 second between captures

    cap.release()
    print(f"\n[DONE] Captured 10 samples for {student_name}.")
    print("Next step: Run 'python3 src/face_encoder.py' to regenerate encodings.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 capture_samples.py <student_name>")
    else:
        name = sys.argv[1].lower()
        capture_samples(name)
