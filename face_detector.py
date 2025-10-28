import os
import cv2
import numpy as np

try:
    import dlib
    _HAS_DLIB = True
except Exception:
    dlib = None
    _HAS_DLIB = False

class FaceDetector:
    def __init__(self):
        # Initialize face detector and landmark predictor
        if not _HAS_DLIB:
            raise RuntimeError("dlib is not installed. Install dlib or set up an alternative face detector.")

        self.face_detector = dlib.get_frontal_face_detector()

        # Allow specifying the predictor path via env var SHAPE_PREDICTOR_PATH
        predictor_path = os.environ.get('SHAPE_PREDICTOR_PATH')
        if predictor_path is None:
            # default to repo root file if present
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            predictor_path = os.path.join(repo_root, 'shape_predictor_68_face_landmarks.dat')

        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"shape predictor not found at: {predictor_path}")

        self.landmark_predictor = dlib.shape_predictor(predictor_path)
        
        # Eye landmarks indices (for 68-point model)
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        self.MOUTH_POINTS = list(range(48, 68))
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        return faces
    
    def extract_eyes(self, frame, face):
        """Extract eye regions from face"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = self.landmark_predictor(gray, face)
        
        left_eye = self.get_eye_region(frame, landmarks, self.LEFT_EYE_POINTS)
        right_eye = self.get_eye_region(frame, landmarks, self.RIGHT_EYE_POINTS)
        
        return left_eye, right_eye
    
    def get_eye_region(self, frame, landmarks, eye_points):
        """Extract and preprocess eye region"""
        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
        
        # Get bounding box with padding
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        # Add padding
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        # Extract eye region
        eye_region = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        if eye_region.size == 0:
            return None
            
        # Convert to grayscale and resize
        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        return eye_gray
    
    def get_eye_landmarks(self, face):
        """Get eye landmarks for EAR calculation"""
        # This would be implemented based on your landmark detection
        pass
    
    def detect_yawn(self, frame, face):
        """Detect yawning using mouth aspect ratio"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = self.landmark_predictor(gray, face)
        
        # Get mouth points
        mouth_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                               for i in self.MOUTH_POINTS])
        
        # Calculate mouth aspect ratio
        A = np.linalg.norm(mouth_points[2] - mouth_points[10])  # vertical distance 1
        B = np.linalg.norm(mouth_points[4] - mouth_points[8])   # vertical distance 2
        C = np.linalg.norm(mouth_points[0] - mouth_points[6])   # horizontal distance
        
        mar = (A + B) / (2.0 * C)
        return mar