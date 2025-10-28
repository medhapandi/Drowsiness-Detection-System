import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    mp = None
    _HAS_MEDIAPIPE = False


class FaceDetector:
    """Face detector using MediaPipe Face Mesh.

    Provides a small compatible API: detect_faces(frame) -> list of face objects,
    extract_eyes(frame, face) -> (left_eye_gray, right_eye_gray),
    detect_yawn(frame, face) -> mouth_aspect_ratio (float)
    """

    # Common landmark groups for eyes and mouth (MediaPipe Face Mesh indices)
    LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_LANDMARKS = [263, 387, 385, 362, 380, 373]
    MOUTH_LANDMARKS = [78, 308, 13, 14, 87, 317]

    def __init__(self, max_num_faces=1, min_detection_confidence=0.5):
        if not _HAS_MEDIAPIPE:
            raise RuntimeError("mediapipe is not installed")

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False,
                                               max_num_faces=max_num_faces,
                                               refine_landmarks=True,
                                               min_detection_confidence=min_detection_confidence)

    def detect_faces(self, frame):
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return []
        return results.multi_face_landmarks

    def extract_eyes(self, frame, face_landmarks):
        h, w = frame.shape[:2]
        lm = face_landmarks.landmark

        def crop_from_indices(indices):
            pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in indices])
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            padding = 4
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            region = frame[y_min:y_max, x_min:x_max]
            if region.size == 0:
                return None
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (24, 24))
            return gray

        left = crop_from_indices(self.LEFT_EYE_LANDMARKS)
        right = crop_from_indices(self.RIGHT_EYE_LANDMARKS)
        return left, right

    def detect_yawn(self, frame, face_landmarks):
        # Compute a simple mouth aspect ratio using selected mouth landmarks
        h, w = frame.shape[:2]
        lm = face_landmarks.landmark
        pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in self.MOUTH_LANDMARKS])
        if pts.shape[0] < 6:
            return 0.0

        # vertical distances
        A = np.linalg.norm(pts[2] - pts[3])
        B = np.linalg.norm(pts[4] - pts[5])
        # horizontal distance
        C = np.linalg.norm(pts[0] - pts[1])
        if C == 0:
            return 0.0
        mar = (A + B) / (2.0 * C)
        return mar
