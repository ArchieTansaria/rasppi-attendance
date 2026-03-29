import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STUDENTS_DIR = os.path.join(DATA_DIR, "students")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Files
ENCODINGS_FILE = os.path.join(DATA_DIR, "encodings.pkl")
ATTENDANCE_LOG_FILE = os.path.join(LOGS_DIR, "attendance.csv")
MEDIAPIPE_MODEL_PATH = os.path.join(DATA_DIR, "face_detector.tflite")

# Face Recognition Settings
TOLERANCE = 0.5 
MATCH_THRESHOLD = 0.75  # Balanced threshold for HOG features
FRAME_RESIZE_FACTOR = 0.5 
PROCESS_ALTERNATE_FRAMES = True 
FRAME_SKIP = 2 
FACE_SIZE = (96, 96) 
MAX_FACES_PER_FRAME = 2 
TEMPORAL_WINDOW = 7  # Increased for smoother identification (needs 4 stable frames)

# Networking Settings
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8000
PI_IP_ADDRESS = "localhost"  # Change to your Pi's IP when running on laptop
